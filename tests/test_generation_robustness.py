"""
Tests for explicit failure handling (BUG-05) and reproducible augmentation (BUG-06).

Only expected failures are counted as discards (Lundeby noise criterion, failed TR
analysis, unrealizable DRR); programming errors surface. Augmented RIRs are built once
per RIR from seeded generators, so the same config always yields the same database.
"""

import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from code.data_reader import GENERATION_SUMMARY, read_dataset
from code.generate_database import DataBaseGenerator
from code.parameters_calculation import (TAE, DrrAugmentationError, TrAugmentationError,
                                         drr_aug, tr_augmentation)

FS = 16000
BANDS = [500, 1000, 4000]


def make_generator(seed=2222, to_augmentate=(), tr_aug=(0.5, 2.1, 0.25), drr_aug_params=(0, 13, 6),
                   rirs_train=('room.wav',), speech_train=()):
    return DataBaseGenerator(
        list(speech_train), [], list(rirs_train), list(to_augmentate), list(rirs_train), [],
        BANDS, 'octave band', FS, -60, 4, False, [0, 40], list(tr_aug), list(drr_aug_params),
        seed=seed,
    )


def measured_like_rir(rt=1.0, duration=2.0, delay_s=0.03, noise_db=-80.0, seed=0):
    """Pre-delay + exponentially decaying noise + a low background noise floor."""
    rng = np.random.default_rng(seed)
    n = int(duration * FS)
    t = np.arange(n) / FS
    rir = rng.standard_normal(n) * np.exp(-6.908 * t / rt)
    rir[0] = 4.0  # clear direct sound
    rir = np.concatenate([np.zeros(int(delay_s * FS)), rir])
    rir += 10 ** (noise_db / 20) * rng.standard_normal(rir.size)
    return rir / np.max(np.abs(rir))


@pytest.fixture
def in_tmp_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the generator creates ./cache and reads ./data
    return tmp_path


# --------------------------------------------------------------------------- #
# Speech length (BUG-05: short files were silently dropped)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("duration_s", [4.05, 4.69, 5.0, 6.3])
def test_speech_is_always_exactly_five_seconds(in_tmp_dir, caplog, duration_s):
    rng = np.random.default_rng(1)
    path = in_tmp_dir / 'speech.wav'
    sf.write(path, 0.1 * rng.standard_normal(int(duration_s * FS)), FS)

    with caplog.at_level(logging.WARNING):
        speech = make_generator()._load_speech(path)

    assert speech.shape == (5 * FS,)
    assert np.max(np.abs(speech)) == pytest.approx(1.0)
    if duration_s < 5.0:
        assert np.all(speech[int(duration_s * FS) + 1:] == 0.0)
        assert 'zero-padding' in caplog.text
    else:
        assert 'zero-padding' not in caplog.text


def test_tae_rejects_wrong_length_instead_of_asserting(in_tmp_dir):
    gen = make_generator()
    with pytest.raises(ValueError, match="exactly 5 s"):
        TAE(np.random.default_rng(0).standard_normal(4 * FS), FS, gen.sos_lowpass_filter)


def test_short_speech_files_produce_records_end_to_end(in_tmp_dir):
    rng = np.random.default_rng(2)
    rir_dir = in_tmp_dir / 'data' / 'RIRs'
    speech_dir = in_tmp_dir / 'data' / 'Speech' / 'train'
    rir_dir.mkdir(parents=True)
    speech_dir.mkdir(parents=True)
    sf.write(rir_dir / 'room.wav', measured_like_rir(), FS)
    sf.write(speech_dir / 'F5s3.wav', 0.1 * rng.standard_normal(int(4.69 * FS)), FS)
    sf.write(speech_dir / 'M1s3.wav', 0.1 * rng.standard_normal(5 * FS), FS)

    gen = make_generator(speech_train=('train/F5s3.wav', 'train/M1s3.wav'))
    records, discards = gen.process_single_rir('room.wav')

    speech_in_records = {r['ReverbedAudio'].split('|')[0] for r in records}
    assert speech_in_records == {'F5s3', 'M1s3'}
    assert len(records) == 2 * len(BANDS)
    assert discards == Counter()


# --------------------------------------------------------------------------- #
# Expected failures vs. bugs
# --------------------------------------------------------------------------- #

def test_noisy_band_rirs_are_counted_as_discards(in_tmp_dir):
    gen = make_generator()
    noisy_rir = measured_like_rir(noise_db=-30.0)  # fails the -60 dB Lundeby criterion
    speech = np.random.default_rng(3).standard_normal(5 * FS)
    discards = Counter()

    records = gen._process_entry(speech, noisy_rir, 'M1s3', 'room', 'train', 'original', discards)

    assert len(records) + discards['band:lundeby_noise'] == len(BANDS)
    assert discards['band:lundeby_noise'] > 0


def test_unrealizable_drr_raises_a_specific_error():
    rir = measured_like_rir()
    out = drr_aug(rir, FS, 6.0)
    assert out.shape == rir.shape and np.max(np.abs(out)) == pytest.approx(1.0)
    with pytest.raises(DrrAugmentationError):
        drr_aug(rir, FS, -60.0)


def test_tr_augmentation_keeps_the_cause_and_does_not_hide_bugs(in_tmp_dir):
    gen = make_generator()
    with pytest.raises(TrAugmentationError) as info:
        tr_augmentation(np.zeros(FS), FS, 1.0, gen.bp_filter)  # all-zero RIR: numeric failure
    assert info.value.cause is not None

    with pytest.raises(AttributeError):  # programming error: no filter bank
        tr_augmentation(measured_like_rir(), FS, 1.0, None)


# --------------------------------------------------------------------------- #
# Reproducible augmentation (BUG-06)
# --------------------------------------------------------------------------- #

def test_tr_augmentation_is_reproducible_with_a_seeded_rng(in_tmp_dir):
    gen = make_generator()
    rir = measured_like_rir()
    a = tr_augmentation(rir, FS, 1.5, gen.bp_filter, rng=np.random.default_rng(10))
    b = tr_augmentation(rir, FS, 1.5, gen.bp_filter, rng=np.random.default_rng(10))
    c = tr_augmentation(rir, FS, 1.5, gen.bp_filter, rng=np.random.default_rng(11))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_augmented_variants_are_reproducible_and_use_five_drr_tr_variants(in_tmp_dir):
    rir = measured_like_rir()

    def variants(seed):
        gen = make_generator(seed=seed, to_augmentate=('room.wav',))
        discards = Counter()
        return gen._augmented_variants('room', rir, discards), discards, gen

    first, discards, gen = variants(2222)
    second, _, _ = variants(2222)
    other_seed, _, _ = variants(7)

    assert [tag for tag, _ in first] == [tag for tag, _ in second]
    for (_, a), (_, b) in zip(first, second):
        np.testing.assert_array_equal(a, b)
    assert any(not np.array_equal(a, b) for (_, a), (_, b) in zip(first, other_seed))

    # This RIR is well behaved: every TR variant succeeds, and exactly 5 of them get the full
    # DRR sweep (v0 picked a random number of them).
    tr_tags = [tag for tag, _ in first if tag.startswith('TR_var_')]
    drr_tags = [tag for tag, _ in first if '_DRR_var_' in tag]
    drr_tr_values = {tag.split('_')[1] for tag in drr_tags}
    assert len(tr_tags) == len(gen.tr_variations)
    assert len(drr_tr_values) == 5
    assert len(drr_tags) + discards['drr_variant:unrealizable'] == 5 * len(gen.drr_variations)


def test_non_selected_and_synthetic_rirs_are_not_augmented(in_tmp_dir):
    gen = make_generator(to_augmentate=('great_hall_x00y00.wav',))
    rir = measured_like_rir()
    assert gen._augmented_variants('great_hall_x00y01', rir, Counter()) == []
    assert gen._augmented_variants('sintetica_Seed1_Tr1.0', rir, Counter()) == []
    # Exact stem match, not substring: a prefix of a selected name is not selected.
    assert gen._augmented_variants('great_hall_x00y0', rir, Counter()) == []


# --------------------------------------------------------------------------- #
# Reading a database folder
# --------------------------------------------------------------------------- #

def test_read_dataset_reads_only_partitions_in_numeric_order(in_tmp_dir):
    db_dir = in_tmp_dir / 'cache' / 'db'
    db_dir.mkdir(parents=True)
    for idx in (0, 1, 2, 10):
        pd.DataFrame({'band': [1000, 500], 'type_data': ['train', 'train'], 'idx': [idx, idx]}) \
            .to_pickle(db_dir / f'{idx}.pkl')
    (db_dir / GENERATION_SUMMARY).write_text('{}', encoding='utf-8')

    db = read_dataset(1000, 'db', sample_frac=1.0, random_state=None, type_data='train')
    assert sorted(db['idx'].tolist()) == [0, 1, 2, 10]
    assert set(db['band']) == {1000}
