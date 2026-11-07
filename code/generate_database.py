import hashlib
import json
import os
import random
from datetime import datetime
from math import nan
from pathlib import Path

import numpy as np
import pandas as pd
from librosa import load
from scipy.signal import butter

from .convolution import causal_convolve
from .cpp import audio_processing
from .mixing import mix_at_snr
from .random_streams import RNG_STREAMS, item_rng  # noqa: F401 (RNG_STREAMS re-exported)
from filters import BandpassFilter
from .parameters_calculation import (TAE, NoiseError,
                                    TrAugmentationError,
                                    colored_noise, drr_aug, find_onset, get_DRR,
                                    tr_augmentation, tr_lundeby)

# Version of the data-generation pipeline. Bump it whenever a change alters the
# generated features (TAE) or labels (descriptors), so that a stale cache built by
# older code is never reused. Version 0 = thesis pipeline (cache names without suffix).
#   1: causal convolution instead of mode='same' (BUG-01).
#   2: C50/C80/D50 integrated from the direct-sound onset (BUG-02).
#   3: noise mixed broadband before the filter bank; one SNR per recording (BUG-03).
#   4: random noise color, clean fraction and per-recording seeded noise (DATA-09).
DATA_PIPELINE_VERSION = 4


class DataBaseGenerator:
    """
    Generate an acoustic‑descriptor database from speech signals and room
    impulse responses (RIRs).

    This class orchestrates the following workflow:

    1. Load and peak‑normalize speech and RIR audio.
    2. Convolve speech with each (original and optionally augmented) RIR.
    3. Split signals into frequency bands using a C++ filter bank.
    4. Compute room/acoustic descriptors (T30, C50, C80, D50, DRR) per band.
    5. Optionally add colored noise (reproducible per recording) and compute TAE features.
    6. Accumulate records suitable for DataFrame / pickle storage.

    The instance is intentionally pickle‑friendly to support multiprocessing;
    heavy C++ filter objects are re‑instantiated in worker processes.
    """

    __slots__ = (
        'speech_files_train', 'speech_files_test', 'rir_files', 'to_augmentate',
        'rirs_for_training', 'rirs_for_testing', 'bands', 'filter_type', 'fs',
        'max_ruido_dB', 'order', 'add_noise', 'snr', 'tr_aug_params', 'drr_aug_params',
        'db_name', 'sos_lowpass_filter', 'tr_variations', 'drr_variations', 'bp_filter',
        'data_path', 'cache_path', 'noise_beta', 'clean_fraction', 'seed'
    )

    def __init__(self, speech_files_train, speech_files_test, rir_files, to_augmentate,
                 rirs_for_training, rirs_for_testing, bands, filter_type, fs,
                 max_ruido_dB, order, add_noise, snr, tr_aug_params, drr_aug_params,
                 noise_beta=(1.0, 1.0), clean_fraction=0.0, seed=0):
        """
        Noise parameters (used only when ``add_noise`` is True; see DATA-09)
        --------------------------------------------------------------------
        snr : [low, high]
            Broadband SNR range in dB, drawn uniformly per recording.
        noise_beta : [low, high], default=(1.0, 1.0)
            Range of the noise spectral exponent (1/f^beta), drawn uniformly per recording.
            (1.0, 1.0) = always pink (legacy behavior).
        clean_fraction : float in [0, 1], default=0.0
            Probability that a recording is left noise-free.
        seed : int, default=0
            Project-wide seed (the config's ``seed``). Each random process derives its own
            stream from it (see ``code/random_streams.py``). For noise, the seed and the recording name
            fully determine whether a recording is noisy, its color, its SNR and the noise
            realization, so the same configuration always yields the same noisy database,
            regardless of processing order or of which other files are included.
        """
        if len(snr) != 2 or snr[0] > snr[-1]:
            raise ValueError(f"`snr` must be [low, high] with low <= high, got {snr}")
        if len(noise_beta) != 2 or noise_beta[0] > noise_beta[1]:
            raise ValueError(f"`noise_beta` must be [low, high] with low <= high, got {noise_beta}")
        if not 0.0 <= clean_fraction <= 1.0:
            raise ValueError(f"`clean_fraction` must be in [0, 1], got {clean_fraction}")
        # --- Parameter assignment ---
        self.speech_files_train = speech_files_train
        self.speech_files_test = speech_files_test
        self.rir_files = rir_files
        self.to_augmentate = to_augmentate
        self.rirs_for_training = rirs_for_training
        self.rirs_for_testing = rirs_for_testing
        self.bands = bands
        self.filter_type = filter_type
        self.fs = fs
        self.max_ruido_dB = max_ruido_dB
        self.order = order
        self.add_noise = add_noise
        self.snr = snr
        self.tr_aug_params = tr_aug_params
        self.drr_aug_params = drr_aug_params
        self.noise_beta = (float(noise_beta[0]), float(noise_beta[1]))
        self.clean_fraction = float(clean_fraction)
        self.seed = int(seed)

        # --- Paths ---
        self.data_path = Path('data')
        self.cache_path = Path('cache')
        self.cache_path.mkdir(exist_ok=True)  # Ensure the cache directory exists.

        # --- Database name template (used for cache filenames) ---
        self.db_name = (
            f'data_base_{self.max_ruido_dB}_noise_{self.add_noise}_'
            f'traug_{"_".join(map(str, self.tr_aug_params))}_'
            f'drraug_{"_".join(map(str, self.drr_aug_params))}_'
            f'snr_{self.snr[0]}_{self.snr[-1]}_'
            f'beta_{self.noise_beta[0]}_{self.noise_beta[1]}_'
            f'clean_{self.clean_fraction}_seed_{self.seed}_'
            f'files_{self._file_lists_digest()}_'
            f'dv{DATA_PIPELINE_VERSION}'
        )

        # --- Pre‑compute supporting filters / variation ranges ---
        cutoff = 20  # Hz
        self.sos_lowpass_filter = butter(self.order, cutoff, fs=self.fs, btype='lowpass', output='sos')
        self.tr_variations = np.arange(self.tr_aug_params[0], self.tr_aug_params[1], self.tr_aug_params[2])
        self.drr_variations = np.arange(self.drr_aug_params[0], self.drr_aug_params[1], self.drr_aug_params[2])
        self.bp_filter = BandpassFilter(self.filter_type, self.fs, self.order, self.bands)  
    
    def _load_and_normalize_audio(self, file_path: Path, duration: float = 5.0) -> np.ndarray:
        """Load an audio file (Librosa) at the instance sample rate and peak‑normalize."""
        audio_data, _ = load(file_path, sr=self.fs, duration=duration)
        max_val = np.max(np.abs(audio_data))
        return audio_data / max_val if max_val > 0 else audio_data

    def _file_lists_digest(self) -> str:
        """
        Short digest of the file lists and of the train/test assignment.

        The split is baked into the database (``type_data``) at generation time, so two
        configs that differ only in their split (e.g. leave-one-room-out folds) must never
        share a cache (BUG-04 / CODE-05).
        """
        payload = json.dumps(
            {
                'rirs_train': sorted(self.rirs_for_training),
                'rirs_test': sorted(self.rirs_for_testing),
                'speech_train': sorted(self.speech_files_train),
                'speech_test': sorted(self.speech_files_test),
                'augment': sorted(self.to_augmentate),
            },
            sort_keys=True,
        )
        return hashlib.sha1(payload.encode('utf-8')).hexdigest()[:10]

    def _speech_file_path(self, split_dir: Path, speech_file: str) -> Path:
        """
        Resolve a speech file entry from the config.

        Entries with a folder (``'train/F1s3.wav'``, from ``splits.list_speech_files``) are
        relative to ``data/Speech``, so a speaker-based split can pool both folders (BUG-04).
        Bare names (``'F1s3.wav'``, v0 configs) are looked up in the split's own folder.
        """
        if Path(speech_file).parent != Path('.'):
            return self.data_path / 'Speech' / speech_file
        return split_dir / speech_file

    def _calculate_descriptors(self, rir_band: np.ndarray, rir_band_from_onset: np.ndarray) -> dict:
        """
        Compute acoustic descriptors (T30, C50, C80, D50, DRR) for a single band.

        Parameters
        ----------
        rir_band : ndarray
            Band-filtered RIR, including any pre-delay. Used for T30 (Lundeby windows
            from the band peak) and DRR, which keep their original definitions.
        rir_band_from_onset : ndarray
            Band-filtered RIR trimmed at the broadband direct-sound onset. Used for the
            energy ratios C50, C80 and D50, which ISO 3382-1 integrates from the direct
            sound (BUG-02).
        """
        t30, _, _ = tr_lundeby(rir_band, self.fs, self.max_ruido_dB)
        c50 = audio_processing.ClarityCalculator.calculate(50, rir_band_from_onset, self.fs)
        c80 = audio_processing.ClarityCalculator.calculate(80, rir_band_from_onset, self.fs)
        d50 = audio_processing.DefinitionCalculator.calculate(rir_band_from_onset, self.fs)
        drr, _, _ = get_DRR(rir_band, self.fs)
        return {'T30': t30, 'C50': c50, 'C80': c80, 'D50': d50, 'DRR': drr}
        
    def _noise_rng(self, recording_name: str) -> np.random.Generator:
        """
        Random generator for the noise of one recording.

        Seeded with ``(seed, RNG_STREAMS['noise'], crc32(recording_name))``: the draws depend
        only on the project seed and on the recording identity (``speech|rir|augmentation``),
        not on processing order, worker scheduling or which other files are in the database.
        See ``code/random_streams.py``.
        """
        return item_rng(self.seed, 'noise', recording_name)

    def _draw_noise_settings(self, rng: np.random.Generator) -> tuple:
        """
        Draw the noise settings of one recording, in a fixed order.

        Returns
        -------
        tuple
            ``(beta, snr_db)``, or ``None`` if the recording stays noise-free (probability
            ``clean_fraction``).
        """
        # The order of the draws is part of the reproducibility contract: do not reorder.
        is_clean = rng.random() < self.clean_fraction
        beta = float(rng.uniform(*self.noise_beta))
        snr_db = float(rng.uniform(self.snr[0], self.snr[-1]))
        return None if is_clean else (beta, snr_db)

    def _build_input_bands(self, reverbed_audio: np.ndarray, rng=None) -> tuple:
        """
        Emulate the recording and split it into bands, as inference does.

        If ``add_noise`` is enabled and the recording is not drawn as clean, colored noise
        (``1/f^beta``) is mixed **broadband** with the reverberant speech at a random SNR,
        and only then is the mixture passed through the filter bank (BUG-03). Color, SNR
        and the clean/noisy decision are drawn per recording (DATA-09).

        Parameters
        ----------
        reverbed_audio : ndarray, shape (n_samples,)
            Clean reverberant speech (speech * RIR).
        rng : numpy.random.Generator or None
            Random source for this recording (see :meth:`_noise_rng`). Required when
            ``add_noise`` is True.

        Returns
        -------
        input_bands : ndarray, shape (n_bands, n_samples)
            Band-filtered recording (speech, plus noise if any).
        snr_db : float
            Broadband SNR in dB, or NaN for a noise-free recording.
        band_snr_db : ndarray, shape (n_bands,)
            Resulting SNR per band in dB (diagnostic), or NaN for a noise-free recording.
        beta : float
            Spectral exponent of the added noise, or NaN for a noise-free recording.
        """
        settings = None
        if self.add_noise:
            if rng is None:
                raise ValueError("`rng` is required when add_noise is True (see _noise_rng)")
            settings = self._draw_noise_settings(rng)

        if settings is None:
            input_bands = self.bp_filter.filter_signals(reverbed_audio.astype(np.float32))
            return input_bands, nan, np.full(len(self.bands), nan), nan

        beta, snr_db = settings
        noise = colored_noise(len(reverbed_audio), beta, rng=rng)
        mixture, gain = mix_at_snr(reverbed_audio, noise, snr_db)

        input_bands = self.bp_filter.filter_signals(mixture.astype(np.float32))
        # Diagnostic per-band SNR: the filter bank is linear, so the speech part of each
        # band is the band-filtered mixture minus the band-filtered noise.
        noise_bands = self.bp_filter.filter_signals((gain * noise).astype(np.float32))
        speech_power = np.mean((input_bands - noise_bands) ** 2, axis=1)
        noise_power = np.mean(noise_bands**2, axis=1)
        band_snr_db = 10 * np.log10(speech_power / noise_power)
        return input_bands, snr_db, band_snr_db, beta

    def process_single_rir(self, rir_file: str) -> list:
        """
        Worker method for multiprocessing.

        Process a *single* RIR across all speech files (train/test split honored),
        apply configured augmentations (TR / nested DRR), compute descriptors and
        TAE features for each band, and return a list of record dictionaries.

        Parameters
        ----------
        rir_file : str
            RIR filename (relative to ``data/RIRs``).

        Returns
        -------
        list of dict
            One record per (speech_file, band, augmentation) combination.
        """
        print(f"Starting process for RIR: {rir_file}...")
    
        # --- Per‑RIR initialization ---
        rir_name = Path(rir_file).stem
        rir_data = self._load_and_normalize_audio(self.data_path / 'RIRs' / rir_file)
        records_for_this_rir = []
        random.seed(int(datetime.now().timestamp()))  # Re‑seed RNG per process.

        # --- Determine train/test membership ---
        if rir_file in self.rirs_for_training:
            speech_files = self.speech_files_train
            speech_type = 'train'
            speech_path = self.data_path / 'Speech' / 'train'
        else:
            speech_files = self.speech_files_test
            speech_type = 'test'
            speech_path = self.data_path / 'Speech' / 'test'

        # --- Iterate over all speech files for this split ---
        for speech_file in speech_files:
            speech_name = Path(speech_file).stem
            speech_data = self._load_and_normalize_audio(
                self._speech_file_path(speech_path, speech_file), duration=5.0
            )

            # 1. Always process the *original* RIR.
            records_for_this_rir.extend(self._process_entry(
                speech_data, rir_data, speech_name, rir_name, speech_type, 'original'
            ))

            # 2. Check if this RIR is subject to augmentation.
            should_augment = not ('sintetica' in rir_name or not any(rir_name in s for s in self.to_augmentate))
            if not should_augment:
                continue
            
            # 3. TR augmentation sweep.
            for tr_var in self.tr_variations:
                try:
                    rir_tr_aug = tr_augmentation(rir_data, self.fs, tr_var, self.bp_filter)
                    rir_tr_aug /= np.max(np.abs(rir_tr_aug))
                    
                    aug_tag = f'TR_var_{tr_var:.2f}'
                    records_for_this_rir.extend(self._process_entry(
                        speech_data, rir_tr_aug, speech_name, rir_name, speech_type, aug_tag
                    ))

                    # 4. Nested DRR augmentation (random subset of TR variants).
                    if tr_var in random.sample(list(self.tr_variations), k=5):
                        for drr_var in self.drr_variations:
                            try:
                                rir_drr_aug = drr_aug(rir_tr_aug, self.fs, drr_var)
                                rir_drr_aug /= np.max(np.abs(rir_drr_aug))
                                
                                aug_tag = f'TR_{tr_var:.2f}_DRR_var_{drr_var:.2f}'
                                records_for_this_rir.extend(self._process_entry(
                                    speech_data, rir_drr_aug, speech_name, rir_name, speech_type, aug_tag
                                ))
                            except Exception:
                                continue
                except (TrAugmentationError, Exception):
                    continue
        
        print(f"Process for RIR: {rir_file} finished. ✅")
        return records_for_this_rir
    
    def _process_entry(self, speech_data: np.ndarray, rir_data: np.ndarray,
                       speech_name: str, rir_name: str, speech_type: str, aug_tag: str) -> list:
        """
        Process a single (speech, RIR[, augmentation]) combination and produce
        per‑band database records.

        Parameters
        ----------
        speech_data : ndarray
            Speech audio array.
        rir_data : ndarray
            RIR audio array (original or augmented).
        speech_name : str
            Base name of the speech file (no extension).
        rir_name : str
            Base name of the RIR file (no extension).
        speech_type : {'train', 'test'}
            Dataset split label.
        aug_tag : str
            Augmentation tag (e.g., ``'original'``, ``'TR_var_0.50'``).

        Returns
        -------
        list of dict
            Records suitable for DataFrame construction.
        """
        # Causal convolution truncated to the speech length, emulating a recording window
        # (BUG-01: mode='same' shifted the output by (len(rir)-1)//2 samples).
        reverbed_audio = causal_convolve(speech_data, rir_data)
        reverbed_audio /= np.max(np.abs(reverbed_audio))

        name = f'{speech_name}|{rir_name}|{aug_tag}'

        # One recording per entry: broadband noise (if enabled), then the filter bank (BUG-03).
        # Noise draws are seeded per recording, so they are reproducible (DATA-09).
        rng = self._noise_rng(name) if self.add_noise else None
        input_bands, snr_db, band_snr_db, noise_beta = self._build_input_bands(reverbed_audio, rng)
        filtered_rir_bands = self.bp_filter.filter_signals(rir_data.astype(np.float32))
        # Energy ratios are integrated from the broadband direct-sound onset, shared by
        # all bands (BUG-02). The RIR is trimmed before filtering.
        onset = find_onset(rir_data)
        filtered_rir_bands_from_onset = self.bp_filter.filter_signals(
            rir_data[onset:].astype(np.float32)
        )

        entry_records = []

        for i, band in enumerate(self.bands):
            try:
                descriptors = self._calculate_descriptors(
                    filtered_rir_bands[i], filtered_rir_bands_from_onset[i]
                )
                tae = TAE(input_bands[i], self.fs, self.sos_lowpass_filter)

                record = {
                    'ReverbedAudio': name,
                    'type_data': speech_type,
                    'band': band,
                    'descriptors': [descriptors['T30'], descriptors['C50'], descriptors['C80'], descriptors['D50']],
                    'drr': descriptors['DRR'],
                    'tae': list(tae),
                    'snr': snr_db,                    # broadband SNR of the recording (dB); NaN = clean
                    'snr_band': float(band_snr_db[i]),  # resulting SNR in this band (dB); NaN = clean
                    'noise_beta': noise_beta,         # noise color exponent (1/f^beta); NaN = clean
                }
                entry_records.append(record)
            except (ValueError, NoiseError, Exception) as e:
                # print(f"Error processing {name} in band {band}: {e}")
                continue
        return entry_records

    def generate_database(self):
        """
        Generate and cache the full database (single‑process version).

        This method mirrors :meth:`process_single_rir` but executes serially
        under the caller's control (no multiprocessing). Results are cached as
        a pickle for reuse.

        Returns
        -------
        pandas.DataFrame or None
            The generated database, or ``None`` if no records were produced.
        """
        # --- Cache check ---
        db_file = self.cache_path / f'{self.db_name}.pkl'
        if db_file.exists():
            print(f"Database already exists at: {db_file}")
            return pd.read_pickle(db_file)

        all_records = []
        
        # --- Main loop over RIRs ---
        for rir_file in self.rir_files:
            print(f"Processing RIR: {rir_file}...")
            rir_name = Path(rir_file).stem
            rir_data = self._load_and_normalize_audio(self.data_path / 'RIRs' / rir_file)

            # Determine train/test membership (avoid code duplication).
            if rir_file in self.rirs_for_training:
                speech_files = self.speech_files_train
                speech_type = 'train'
                speech_path = self.data_path / 'Speech' / 'train'
            else:
                speech_files = self.speech_files_test
                speech_type = 'test'
                speech_path = self.data_path / 'Speech' / 'test'

            # Loop over all speech files for this split.
            for speech_file in speech_files:
                speech_name = Path(speech_file).stem
                speech_data = self._load_and_normalize_audio(
                    self._speech_file_path(speech_path, speech_file), duration=5.0
                )

                # 1. Always process the original RIR.
                all_records.extend(self._process_entry(
                    speech_data, rir_data, speech_name, rir_name, speech_type, 'original'
                ))

                # 2. If RIR is eligible for augmentation.
                should_augment = not ('sintetica' in rir_name or not any(rir_name in s for s in self.to_augmentate))
                if not should_augment:
                    continue
                
                # 3. TR augmentation sweep.
                for tr_var in self.tr_variations:
                    try:
                        rir_tr_aug = tr_augmentation(rir_data, self.fs, tr_var, self.bp_filter)
                        rir_tr_aug /= np.max(np.abs(rir_tr_aug))
                        
                        aug_tag = f'TR_var_{tr_var:.2f}'
                        all_records.extend(self._process_entry(
                            speech_data, rir_tr_aug, speech_name, rir_name, speech_type, aug_tag
                        ))

                        # 4. Nested DRR augmentation (as in the multiprocessing version).
                        if tr_var in random.sample(list(self.tr_variations), k=5):
                            for drr_var in self.drr_variations:
                                try:
                                    rir_drr_aug = drr_aug(rir_tr_aug, self.fs, drr_var)
                                    rir_drr_aug /= np.max(np.abs(rir_drr_aug))
                                    
                                    aug_tag = f'TR_{tr_var:.2f}_DRR_var_{drr_var:.2f}'
                                    all_records.extend(self._process_entry(
                                        speech_data, rir_drr_aug, speech_name, rir_name, speech_type, aug_tag
                                    ))
                                except Exception as e:
                                    # print(f"DRR augmentation error {drr_var}: {e}")
                                    continue
                    except (TrAugmentationError, Exception) as e:
                        # print(f"TR augmentation error {tr_var}: {e}")
                        continue
        
        # --- DataFrame creation & save ---
        if not all_records:
            print("No records were generated. The database is empty.")
            return None
            
        final_db = pd.DataFrame(all_records)
        final_db.to_pickle(db_file)
        print(f"Database generated and saved at: {db_file}")
        
        return final_db