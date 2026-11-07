"""
Shared configuration for the v1 experiments (corrected pipeline, data pipeline v4+).

Experiment configs build their namespace with :func:`make_config`, e.g.::

    from configs.v1_base import make_config
    globals().update(make_config(exp_num=5, add_noise=False))

Differences from the v0 (thesis) configs ``exp1.py`` … ``exp4.py``:

* **Group-aware splits** (BUG-04): real RIRs are split by **room** (whole rooms are held
  out), synthetic RIRs per file, speech by **speaker**, and validation by source RIR
  (see ``code/splits.py``). The development split holds out ``classroom``; milestone
  baselines run leave-one-room-out folds (one config per held-out room).
* Test speakers ``F4``, ``M4``, ``M9`` (8 of 38 files); the original train/test speech
  folders are pooled.
* Sorted file listings and seed-derived random choices: the same config always yields the
  same split, augmentation choice and noise (BUG-06, DATA-09).
* Noisy variant (DATA-09): broadband SNR 0–40 dB, noise color β ∈ [1, 2] (pink to
  brown) and 20 % noise-free recordings.
* Lundeby noise criterion −45 dB (DATA-10): the ISO 3382-1 minimum dynamic range for T30.
  The thesis' −60 dB kept no real RIR at 125/250 Hz and only ~60 % at 500 Hz–1 kHz.

The decisions behind these values are recorded in ``docs/improvements/``.
"""

from code.splits import (list_rir_files, list_speech_files, select_rirs_to_augment,
                         split_rirs, split_speech)

SEED = 2222
DEV_TEST_ROOMS = ('classroom',)
TEST_SPEAKERS = ('F4', 'M4', 'M9')


def make_config(exp_num, add_noise, test_rooms=DEV_TEST_ROOMS, max_ruido_dB=-45):
    """
    Build the full configuration dictionary of a v1 experiment.

    Parameters
    ----------
    exp_num : int or str
        Experiment identifier; results go to ``results/exp<exp_num>``.
    add_noise : bool
        Whether to build the noisy variant of the database.
    test_rooms : sequence of str, default=('classroom',)
        Real rooms held out for testing (one per leave-one-room-out fold).
    max_ruido_dB : float, default=-45
        Lundeby noise acceptance criterion for the RIR descriptors (DATA-10).
    """
    seed = SEED

    # --- Data and splits (BUG-04) ---
    files_rirs = list_rir_files('data/RIRs')
    rirs_for_training, rirs_for_testing = split_rirs(
        files_rirs, test_rooms=list(test_rooms), synthetic_test_fraction=0.2, seed=seed
    )
    to_augmentate = select_rirs_to_augment(files_rirs, per_room=15, seed=seed)
    files_speech_train, files_speech_test = split_speech(
        list_speech_files('data/Speech'), test_speakers=list(TEST_SPEAKERS)
    )

    return dict(
        seed=seed,
        exp_num=exp_num,
        test_rooms=list(test_rooms),
        test_speakers=list(TEST_SPEAKERS),

        files_rirs=files_rirs,
        rirs_for_training=rirs_for_training,
        rirs_for_testing=rirs_for_testing,
        to_augmentate=to_augmentate,
        files_speech_train=files_speech_train,
        files_speech_test=files_speech_test,

        # Analysis bands and filtering.
        bands=[125, 250, 500, 1000, 2000, 4000, 8000],
        filter_type='octave band',
        fs=16000,
        order=4,
        max_ruido_dB=max_ruido_dB,

        # Noise (DATA-09). Ignored when add_noise is False.
        add_noise=add_noise,
        snr=[0, 40],
        noise_beta=[1.0, 2.0],
        clean_fraction=0.2,

        # Augmentation sweeps.
        tr_aug=[0.2, 3.1, 0.1],
        drr_aug=[-6, 19, 1],

        # Reading the generated database.
        sample_frac=1.0,
        random_state=seed,

        # Model (same architecture as the thesis, so v1 isolates the data fixes).
        filters=[32, 18, 8, 4],
        kernel_size=[10, 5, 5, 5],
        activation=['relu', 'relu', 'relu', 'relu'],
        pool_size=[2, 2, 2],
        learning_rate=0.001,

        # Training. validation_split is the fraction of training RIRs held out for validation.
        validation_split=0.1,
        batch_size=1024,
        epochs=500,
    )
