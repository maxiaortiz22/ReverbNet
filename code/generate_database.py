import hashlib
import json
import logging
from collections import Counter
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
from .parameters_calculation import (TAE, DrrAugmentationError, NoiseError,
                                    TrAugmentationError,
                                    colored_noise, drr_aug, find_onset, get_DRR,
                                    tr_augmentation, tr_lundeby)

logger = logging.getLogger(__name__)

# Duration of every speech recording fed to the pipeline (the TAE expects 5 s × 40 Hz).
SPEECH_DURATION_S = 5.0

# Short speech files already reported in this process (avoids one warning per RIR).
_WARNED_SHORT_SPEECH: set[str] = set()

# Version of the data-generation pipeline. Bump it whenever a change alters the
# generated features (TAE) or labels (descriptors), so that a stale cache built by
# older code is never reused. Version 0 = thesis pipeline (cache names without suffix).
#   1: causal convolution instead of mode='same' (BUG-01).
#   2: C50/C80/D50 integrated from the direct-sound onset (BUG-02).
#   3: noise mixed broadband before the filter bank; one SNR per recording (BUG-03).
#   4: random noise color, clean fraction and per-recording seeded noise (DATA-09).
#   5: reproducible augmentation built once per RIR, exactly 5 DRR TR-variants per RIR,
#      short speech zero-padded to 5 s, only expected failures discarded (BUG-05/06).
#   6: only training RIRs are augmented; the test set holds original RIRs only (DATA-11).
DATA_PIPELINE_VERSION = 6


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

    def process_single_rir(self, rir_file: str) -> tuple:
        """
        Worker method for multiprocessing.

        Process a *single* RIR across all speech files of its split. The RIR variants
        (original + TR/DRR augmentations) are built **once per RIR** with seeded generators
        and then convolved with every speech file, so every speech file sees exactly the
        same augmented RIRs and labels (BUG-06, CODE-01). Previously the augmentation was
        recomputed for each speech file with a clock-seeded, non-reproducible random tail.

        Parameters
        ----------
        rir_file : str
            RIR filename (relative to ``data/RIRs``).

        Returns
        -------
        records : list of dict
            One record per (speech file, RIR variant, band).
        discards : collections.Counter
            Discarded items by reason, keyed ``'<unit>:<reason>'``: ``band`` (one band record
            of one recording), ``tr_variant`` or ``drr_variant`` (one RIR variant, i.e. all
            its speech files and bands). Only expected failures are counted; anything else
            raises (BUG-05).
        """
        rir_name = Path(rir_file).stem
        rir_data = self._load_and_normalize_audio(self.data_path / 'RIRs' / rir_file)
        discards = Counter()

        # --- Determine train/test membership ---
        if rir_file in self.rirs_for_training:
            speech_files = self.speech_files_train
            speech_type = 'train'
            speech_path = self.data_path / 'Speech' / 'train'
        else:
            speech_files = self.speech_files_test
            speech_type = 'test'
            speech_path = self.data_path / 'Speech' / 'test'

        variants = [('original', rir_data)] + self._augmented_variants(rir_name, rir_data, discards)
        # Descriptors depend only on the RIR variant: compute them once, not per speech file.
        rir_descriptors = [self._rir_band_descriptors(variant) for _, variant in variants]

        records = []
        for speech_file in speech_files:
            speech_name = Path(speech_file).stem
            speech_data = self._load_speech(self._speech_file_path(speech_path, speech_file))
            for (aug_tag, variant), descriptors in zip(variants, rir_descriptors):
                records.extend(self._process_entry(
                    speech_data, variant, speech_name, rir_name, speech_type, aug_tag, discards,
                    rir_descriptors=descriptors,
                ))
        return records, discards

    def _rir_band_descriptors(self, rir_data: np.ndarray) -> list:
        """
        Descriptors of every band of one RIR variant.

        Returns
        -------
        list
            One item per band: the descriptor dict, or the discard reason (``str``) when the
            band is an expected failure (``'lundeby_noise'``: the band RIR does not meet the
            Lundeby noise criterion; ``'non_finite_descriptor'``). Any other exception is a
            bug and propagates (BUG-05).
        """
        filtered_rir_bands = self.bp_filter.filter_signals(rir_data.astype(np.float32))
        # Energy ratios are integrated from the broadband direct-sound onset, shared by
        # all bands (BUG-02). The RIR is trimmed before filtering.
        onset = find_onset(rir_data)
        filtered_rir_bands_from_onset = self.bp_filter.filter_signals(
            rir_data[onset:].astype(np.float32)
        )

        per_band = []
        for i in range(len(self.bands)):
            try:
                descriptors = self._calculate_descriptors(
                    filtered_rir_bands[i], filtered_rir_bands_from_onset[i]
                )
            except NoiseError:
                per_band.append('lundeby_noise')
                continue
            values = [descriptors[k] for k in ('T30', 'C50', 'C80', 'D50')]
            per_band.append(descriptors if np.all(np.isfinite(values)) else 'non_finite_descriptor')
        return per_band

    def _should_augment(self, rir_name: str) -> bool:
        """
        Whether ``rir_name`` (a file stem) is a real **training** RIR selected for augmentation.

        Augmentation is a training device (DATA-11): test RIRs are never augmented, even if a
        config lists them in ``to_augmentate``. In EXP-5 the augmented variants of 13 held-out
        RIRs were 56 % of the test records and dominated the headline metrics.
        """
        if 'sintetica' in rir_name:
            return False
        if rir_name not in {Path(f).stem for f in self.rirs_for_training}:
            return False
        return rir_name in {Path(f).stem for f in self.to_augmentate}

    def _augmented_variants(self, rir_name: str, rir_data: np.ndarray, discards: Counter) -> list:
        """
        Build the TR and nested DRR variants of one RIR, reproducibly.

        * One TR variant per value of the TR sweep; the synthesized noiseless tail uses a
          generator seeded with ``(seed, 'tr_tail', rir_name|tr)``.
        * DRR variants are built for exactly ``min(5, n_tr)`` TR variants, chosen once per RIR
          with ``(seed, 'drr_subset', rir_name)``. (v0 re-drew the subset at every iteration,
          so each TR variant entered with p = 5/29.)

        Expected failures are counted in ``discards``: ``TrAugmentationError`` (by cause,
        see BUG-09) and unrealizable DRR targets.

        Returns
        -------
        list of (str, ndarray)
            ``(augmentation tag, peak-normalized RIR)`` pairs.
        """
        if not self._should_augment(rir_name):
            return []

        n_drr = min(5, len(self.tr_variations))
        drr_subset = set(
            item_rng(self.seed, 'drr_subset', rir_name)
            .choice(len(self.tr_variations), size=n_drr, replace=False)
            .tolist()
        )

        variants = []
        for idx, tr_var in enumerate(self.tr_variations):
            tail_rng = item_rng(self.seed, 'tr_tail', f'{rir_name}|{tr_var:.2f}')
            try:
                rir_tr = tr_augmentation(rir_data, self.fs, tr_var, self.bp_filter, rng=tail_rng)
            except TrAugmentationError as err:
                discards[f'tr_variant:tr_augmentation_{err.cause}'] += 1
                continue
            rir_tr = rir_tr / np.max(np.abs(rir_tr))
            variants.append((f'TR_var_{tr_var:.2f}', rir_tr))

            if idx not in drr_subset:
                continue
            for drr_var in self.drr_variations:
                try:
                    rir_drr = drr_aug(rir_tr, self.fs, drr_var)
                except DrrAugmentationError:
                    discards['drr_variant:unrealizable'] += 1
                    continue
                variants.append((f'TR_{tr_var:.2f}_DRR_var_{drr_var:.2f}', rir_drr / np.max(np.abs(rir_drr))))
        return variants

    def _load_speech(self, file_path: Path) -> np.ndarray:
        """
        Load a speech file as exactly ``SPEECH_DURATION_S`` seconds, peak-normalized.

        Longer files are cropped; shorter files are zero-padded at the end (the speaker
        stops and the recording window continues) with a warning. In v0 the two short
        training files (F5s3, M8s3) were silently dropped by the TAE length check (BUG-05).
        """
        n_samples = int(round(SPEECH_DURATION_S * self.fs))
        audio, _ = load(file_path, sr=self.fs, duration=SPEECH_DURATION_S)
        if audio.size < n_samples:
            if str(file_path) not in _WARNED_SHORT_SPEECH:  # once per file and process
                _WARNED_SHORT_SPEECH.add(str(file_path))
                logger.warning("Speech file %s lasts %.2f s; zero-padding to %.1f s",
                               file_path, audio.size / self.fs, SPEECH_DURATION_S)
            audio = np.pad(audio, (0, n_samples - audio.size))
        audio = audio[:n_samples]
        peak = np.max(np.abs(audio))
        return audio / peak if peak > 0 else audio

    def _process_entry(self, speech_data: np.ndarray, rir_data: np.ndarray,
                       speech_name: str, rir_name: str, speech_type: str, aug_tag: str,
                       discards: Counter | None = None, rir_descriptors: list | None = None) -> list:
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
        discards : collections.Counter, optional
            Incremented with ``'band:<reason>'`` for every discarded band record.
        rir_descriptors : list, optional
            Output of :meth:`_rir_band_descriptors` for ``rir_data``, to avoid recomputing
            it for every speech file (CODE-01). Computed here if omitted.

        Returns
        -------
        list of dict
            Records suitable for DataFrame construction.
        """
        if discards is None:
            discards = Counter()
        if rir_descriptors is None:
            rir_descriptors = self._rir_band_descriptors(rir_data)
        # Causal convolution truncated to the speech length, emulating a recording window
        # (BUG-01: mode='same' shifted the output by (len(rir)-1)//2 samples).
        reverbed_audio = causal_convolve(speech_data, rir_data)
        reverbed_audio /= np.max(np.abs(reverbed_audio))

        name = f'{speech_name}|{rir_name}|{aug_tag}'

        # One recording per entry: broadband noise (if enabled), then the filter bank (BUG-03).
        # Noise draws are seeded per recording, so they are reproducible (DATA-09).
        rng = self._noise_rng(name) if self.add_noise else None
        input_bands, snr_db, band_snr_db, noise_beta = self._build_input_bands(reverbed_audio, rng)

        entry_records = []

        for i, band in enumerate(self.bands):
            descriptors = rir_descriptors[i]
            if isinstance(descriptors, str):  # expected discard, counted per band record
                discards[f'band:{descriptors}'] += 1
                continue
            values = [descriptors['T30'], descriptors['C50'], descriptors['C80'], descriptors['D50']]

            tae = TAE(input_bands[i], self.fs, self.sos_lowpass_filter)
            entry_records.append({
                'ReverbedAudio': name,
                'type_data': speech_type,
                'band': band,
                'descriptors': values,
                'drr': descriptors['DRR'],
                'tae': list(tae),
                'snr': snr_db,                      # broadband SNR of the recording (dB); NaN = clean
                'snr_band': float(band_snr_db[i]),  # resulting SNR in this band (dB); NaN = clean
                'noise_beta': noise_beta,           # noise color exponent (1/f^beta); NaN = clean
            })
        return entry_records

    def generate_database(self):
        """
        Generate and cache the full database serially (no multiprocessing).

        Reuses :meth:`process_single_rir` for every RIR, so it produces exactly the same
        records as the multiprocessing path in ``run.py`` (CODE-09: this used to be a
        duplicated copy of the worker logic).

        Returns
        -------
        pandas.DataFrame or None
            The generated database, or ``None`` if no records were produced.
        """
        db_file = self.cache_path / f'{self.db_name}.pkl'
        if db_file.exists():
            logger.info("Database already exists at: %s", db_file)
            return pd.read_pickle(db_file)

        all_records, discards = [], Counter()
        for rir_file in self.rir_files:
            records, rir_discards = self.process_single_rir(rir_file)
            all_records.extend(records)
            discards.update(rir_discards)

        for reason, count in sorted(discards.items()):
            logger.info("Discarded %s: %d", reason, count)
        if not all_records:
            logger.warning("No records were generated. The database is empty.")
            return None

        final_db = pd.DataFrame(all_records)
        final_db.to_pickle(db_file)
        logger.info("Database generated and saved at: %s", db_file)
        return final_db


def generator_from_config(config: dict) -> DataBaseGenerator:
    """
    Build the database generator described by an experiment config.

    Single place that maps config keys to :class:`DataBaseGenerator` arguments, shared by
    ``run.py`` and the scripts that need the database name of a run (e.g.
    ``scripts/recover_test_metadata.py``), so they can never disagree on the cache.

    Parameters
    ----------
    config : dict
        Namespace of an experiment config (``configs/*.py``).

    Returns
    -------
    DataBaseGenerator
        Generator whose ``db_name`` identifies the run's cache folder.
    """
    return DataBaseGenerator(
        config['files_speech_train'], config['files_speech_test'], config['files_rirs'],
        config['to_augmentate'], config['rirs_for_training'],
        config['rirs_for_testing'], config['bands'], config['filter_type'], config['fs'],
        config['max_ruido_dB'], config['order'], config['add_noise'], config['snr'],
        config['tr_aug'], config['drr_aug'],
        # Noise options (DATA-09). Defaults reproduce the legacy setup: always pink noise,
        # no noise-free fraction. The project-wide seed makes the noise reproducible.
        noise_beta=config.get('noise_beta', [1.0, 1.0]),
        clean_fraction=config.get('clean_fraction', 0.0),
        seed=config['seed'],
    )
