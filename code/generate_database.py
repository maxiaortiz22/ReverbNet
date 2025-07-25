import os
import random
from datetime import datetime
from math import nan
from pathlib import Path

import numpy as np
import pandas as pd
from librosa import load
from scipy.signal import butter, fftconvolve

from .cpp import audio_processing
from filters import BandpassFilter
from .parameters_calculation import (TAE, NoiseError,
                                    TrAugmentationError,
                                    drr_aug, get_DRR, pink_noise,
                                    tr_augmentation, tr_lundeby)


class DataBaseGenerator:
    """
    Generate an acoustic‑descriptor database from speech signals and room
    impulse responses (RIRs).

    This class orchestrates the following workflow:

    1. Load and peak‑normalize speech and RIR audio.
    2. Convolve speech with each (original and optionally augmented) RIR.
    3. Split signals into frequency bands using a C++ filter bank.
    4. Compute room/acoustic descriptors (T30, C50, C80, D50, DRR) per band.
    5. Optionally add pink noise at random SNRs and compute TAE features.
    6. Accumulate records suitable for DataFrame / pickle storage.

    The instance is intentionally pickle‑friendly to support multiprocessing;
    heavy C++ filter objects are re‑instantiated in worker processes.
    """

    __slots__ = (
        'speech_files_train', 'speech_files_test', 'rir_files', 'to_augmentate',
        'rirs_for_training', 'rirs_for_testing', 'bands', 'filter_type', 'fs',
        'max_ruido_dB', 'order', 'add_noise', 'snr', 'tr_aug_params', 'drr_aug_params',
        'db_name', 'sos_lowpass_filter', 'tr_variations', 'drr_variations', 'bp_filter',
        'data_path', 'cache_path'
    )

    def __init__(self, speech_files_train, speech_files_test, rir_files, to_augmentate,
                 rirs_for_training, rirs_for_testing, bands, filter_type, fs,
                 max_ruido_dB, order, add_noise, snr, tr_aug_params, drr_aug_params):
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
        
        # --- Paths ---
        self.data_path = Path('data')
        self.cache_path = Path('cache')
        self.cache_path.mkdir(exist_ok=True)  # Ensure the cache directory exists.

        # --- Database name template (used for cache filenames) ---
        self.db_name = (
            f'data_base_{self.max_ruido_dB}_noise_{self.add_noise}_'
            f'traug_{"_".join(map(str, self.tr_aug_params))}_'
            f'drraug_{"_".join(map(str, self.drr_aug_params))}_'
            f'snr_{self.snr[0]}_{self.snr[-1]}'
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

    def _calculate_descriptors(self, rir_band: np.ndarray) -> dict:
        """Compute acoustic descriptors (T30, C50, C80, D50, DRR) for a single‑band RIR."""
        t30, _, _ = tr_lundeby(rir_band, self.fs, self.max_ruido_dB)
        c50 = audio_processing.ClarityCalculator.calculate(50, rir_band, self.fs)
        c80 = audio_processing.ClarityCalculator.calculate(80, rir_band, self.fs)
        d50 = audio_processing.DefinitionCalculator.calculate(rir_band, self.fs)
        drr, _, _ = get_DRR(rir_band, self.fs)
        return {'T30': t30, 'C50': c50, 'C80': c80, 'D50': d50, 'DRR': drr}
        
    def _get_tae_with_noise(self, reverbed_audio_band: np.ndarray) -> tuple:
        """
        Compute TAE for a speech*RIR band, optionally adding pink noise at random SNR.

        Returns
        -------
        tae : list
            TAE feature vector.
        snr_required : float or nan
            Applied SNR (dB) if noise was added; NaN otherwise.
        """
        if not self.add_noise:
            tae = TAE(reverbed_audio_band, self.fs, self.sos_lowpass_filter)
            return list(tae), nan

        noise_data = pink_noise(len(reverbed_audio_band))
        rms_signal = audio_processing.AudioProcessor.rms(reverbed_audio_band)
        rms_noise = audio_processing.AudioProcessor.rms(noise_data)
        
        snr_required = np.random.uniform(self.snr[0], self.snr[-1])
        comp = audio_processing.AudioProcessor.rms_comp(rms_signal, rms_noise, snr_required)
        noise_data_comp = noise_data * comp
        
        reverbed_noisy_audio = reverbed_audio_band + noise_data_comp
        # Optional normalization could occur here; TAE is generally scale‑robust.
        
        tae = TAE(reverbed_noisy_audio, self.fs, self.sos_lowpass_filter)
        return list(tae), snr_required

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
            speech_data = self._load_and_normalize_audio(speech_path / speech_file, duration=5.0)

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
        reverbed_audio = fftconvolve(speech_data, rir_data, mode='same')
        reverbed_audio /= np.max(np.abs(reverbed_audio))

        filtered_speech_bands = self.bp_filter.filter_signals(reverbed_audio.astype(np.float32))
        filtered_rir_bands = self.bp_filter.filter_signals(rir_data.astype(np.float32))
        
        entry_records = []
        name = f'{speech_name}|{rir_name}|{aug_tag}'

        for i, band in enumerate(self.bands):
            try:
                descriptors = self._calculate_descriptors(filtered_rir_bands[i])
                tae, snr = self._get_tae_with_noise(filtered_speech_bands[i])

                record = {
                    'ReverbedAudio': name,
                    'type_data': speech_type,
                    'band': band,
                    'descriptors': [descriptors['T30'], descriptors['C50'], descriptors['C80'], descriptors['D50']],
                    'drr': descriptors['DRR'],
                    'tae': tae,
                    'snr': snr
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
                speech_data = self._load_and_normalize_audio(speech_path / speech_file, duration=5.0)

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