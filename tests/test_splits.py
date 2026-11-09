"""Tests for group-aware splits and train-only target normalization (BUG-04)."""

import numpy as np
import pytest

from code.model import normalize_descriptors
from code.splits import (ROOMS, SYNTHETIC, recording_rir, rir_group, select_rirs_to_augment,
                         speaker_of, split_rirs, split_speech, validation_mask)

SEED = 2222

REAL_RIRS = (
    [f'great_hall_x{i:02d}y00.wav' for i in range(20)]
    + [f'octagon_x{i:02d}y00.wav' for i in range(20)]
    + [f'classroom_{i:02d}x00y.wav' for i in range(15)]
)
SYNTHETIC_RIRS = [f'sintetica_Seed{i}_Tr{0.2 + (i % 29) / 10:.1f}.wav' for i in range(2000)]
ALL_RIRS = REAL_RIRS + SYNTHETIC_RIRS

SPEECH = sorted(
    [f'train/{s}s3.wav' for s in ('F1', 'F2', 'F5', 'M1', 'M4', 'M9')]
    + [f'train/{s}s5.wav' for s in ('F1', 'F2', 'M1', 'M4')]
    + [f'test/{s}s4.wav' for s in ('F1', 'F2', 'M1', 'M4')]
)


# --------------------------------------------------------------------------- #
# Group keys
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "rir_file, group",
    [
        ('great_hall_x00y05.wav', 'great_hall'),
        ('octagon_x10y00.wav', 'octagon'),
        ('classroom_00x05y.wav', 'classroom'),
        ('sintetica_Seed1036663424_Tr2.0.wav', SYNTHETIC),
    ],
)
def test_rir_group(rir_file, group):
    assert rir_group(rir_file) == group


def test_unknown_rir_room_raises():
    with pytest.raises(ValueError):
        rir_group('new_dataset_room_01.wav')


@pytest.mark.parametrize(
    "speech_file, speaker",
    [('train/F1s3.wav', 'F1'), ('test/M4s4.wav', 'M4'), ('M9s4.wav', 'M9'), ('F12s3.wav', 'F12')],
)
def test_speaker_of(speech_file, speaker):
    assert speaker_of(speech_file) == speaker


def test_unknown_speaker_raises():
    with pytest.raises(ValueError):
        speaker_of('train/announcer.wav')


def test_recording_rir():
    assert recording_rir('F1s3|great_hall_x00y00|TR_var_0.50') == 'great_hall_x00y00'
    with pytest.raises(ValueError):
        recording_rir('F1s3|great_hall_x00y00')


# --------------------------------------------------------------------------- #
# RIR split
# --------------------------------------------------------------------------- #

def test_rooms_never_appear_on_both_sides():
    train, test = split_rirs(ALL_RIRS, test_rooms=['classroom'], synthetic_test_fraction=0.2, seed=SEED)

    assert sorted(train + test) == sorted(ALL_RIRS)
    assert not set(train) & set(test)
    train_rooms = {rir_group(f) for f in train} - {SYNTHETIC}
    test_rooms = {rir_group(f) for f in test} - {SYNTHETIC}
    assert test_rooms == {'classroom'}
    assert train_rooms == {'great_hall', 'octagon'}


def test_synthetic_test_fraction_and_stability():
    _, test = split_rirs(ALL_RIRS, ['classroom'], 0.2, SEED)
    synthetic_test = [f for f in test if rir_group(f) == SYNTHETIC]
    assert len(synthetic_test) / len(SYNTHETIC_RIRS) == pytest.approx(0.2, abs=0.03)

    # Each file's assignment depends only on the seed and its name: a subset of the files,
    # in another order, gets the same assignments.
    subset = list(reversed(SYNTHETIC_RIRS[::3]))
    _, test_subset = split_rirs(subset, [], 0.2, SEED)
    assert set(test_subset) == set(synthetic_test) & set(subset)

    _, other_seed = split_rirs(ALL_RIRS, ['classroom'], 0.2, SEED + 1)
    assert set(other_seed) != set(test)


@pytest.mark.parametrize(
    "test_rooms, fraction",
    [(['library'], 0.2), (list(ROOMS), 0.2), (['classroom'], 1.0), (['classroom'], -0.1)],
)
def test_invalid_rir_split_raises(test_rooms, fraction):
    with pytest.raises(ValueError):
        split_rirs(ALL_RIRS, test_rooms, fraction, SEED)


# --------------------------------------------------------------------------- #
# Speech split
# --------------------------------------------------------------------------- #

def test_speakers_never_appear_on_both_sides():
    train, test = split_speech(SPEECH, test_speakers=['F2', 'M4'])

    assert sorted(train + test) == SPEECH
    assert {speaker_of(f) for f in test} == {'F2', 'M4'}
    assert not {speaker_of(f) for f in train} & {'F2', 'M4'}
    # Files from both original folders are pooled.
    assert 'test/F1s4.wav' in train and 'train/F2s3.wav' in test


@pytest.mark.parametrize("test_speakers", [['X1'], ['F1', 'F2', 'F5', 'M1', 'M4', 'M9']])
def test_invalid_speech_split_raises(test_speakers):
    with pytest.raises(ValueError):
        split_speech(SPEECH, test_speakers)


# --------------------------------------------------------------------------- #
# Augmentation selection
# --------------------------------------------------------------------------- #

def test_select_rirs_to_augment_per_room_and_reproducible():
    selected = select_rirs_to_augment(ALL_RIRS, per_room=5, seed=SEED)

    rooms = [rir_group(f) for f in selected]
    assert {room: rooms.count(room) for room in set(rooms)} == {r: 5 for r in ROOMS}
    assert selected == select_rirs_to_augment(list(reversed(ALL_RIRS)), per_room=5, seed=SEED)
    assert selected != select_rirs_to_augment(ALL_RIRS, per_room=5, seed=SEED + 1)

    with pytest.raises(ValueError):
        select_rirs_to_augment(ALL_RIRS, per_room=16, seed=SEED)  # classroom has 15


# --------------------------------------------------------------------------- #
# Validation mask
# --------------------------------------------------------------------------- #

def make_records(rirs, bands=(125, 1000)):
    names = []
    for rir in rirs:
        stem = rir.removesuffix('.wav')
        for speech in ('F1s3', 'M1s5'):
            for tag in ('original', 'TR_var_0.50'):
                names.extend([f'{speech}|{stem}|{tag}'] * len(bands))
    return names


def test_validation_mask_keeps_whole_rirs_together():
    names = make_records(REAL_RIRS[:40] + SYNTHETIC_RIRS[:400])
    mask = validation_mask(names, fraction=0.1, seed=SEED)

    val_rirs = {recording_rir(n) for n, m in zip(names, mask) if m}
    train_rirs = {recording_rir(n) for n, m in zip(names, mask) if not m}
    assert not val_rirs & train_rirs
    assert len(val_rirs) / (len(val_rirs) + len(train_rirs)) == pytest.approx(0.1, abs=0.04)


def test_validation_mask_is_consistent_across_bands_and_subsets():
    names = make_records(SYNTHETIC_RIRS[:300])
    full = dict(zip(names, validation_mask(names, 0.1, SEED)))

    # A band where some RIRs were discarded must keep the same decision for the rest.
    subset = names[::7]
    for name, flag in zip(subset, validation_mask(subset, 0.1, SEED)):
        assert flag == full[name]


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.2])
def test_invalid_validation_fraction_raises(fraction):
    with pytest.raises(ValueError):
        validation_mask(make_records(SYNTHETIC_RIRS[:5]), fraction, SEED)


# --------------------------------------------------------------------------- #
# Target normalization
# --------------------------------------------------------------------------- #

def test_normalization_uses_training_statistics_only():
    rng = np.random.default_rng(0)
    y_train = rng.uniform(0, 10, size=(500, 4, 1))
    y_val = rng.uniform(0, 10, size=(50, 4, 1))
    y_test = rng.uniform(0, 10, size=(80, 4, 1))

    out = normalize_descriptors(y_train, y_val, y_test)
    out_extreme_test = normalize_descriptors(y_train, y_val, 1000 * y_test)

    expected_norm = np.percentile(y_train[:, :, 0], 95, axis=0)
    np.testing.assert_allclose(out[3:], expected_norm)
    np.testing.assert_allclose(out_extreme_test[3:], expected_norm)  # test data does not leak
    np.testing.assert_allclose(out[0], y_train / expected_norm.reshape(1, 4, 1))
    np.testing.assert_allclose(out[2], y_test / expected_norm.reshape(1, 4, 1))
    assert out[1].shape == y_val.shape


def test_normalization_rejects_wrong_shape():
    with pytest.raises(ValueError):
        normalize_descriptors(np.ones((10, 4)), np.ones((2, 4, 1)), np.ones((2, 4, 1)))


# --------------------------------------------------------------------------- #
# Speech paths in the generator
# --------------------------------------------------------------------------- #

def test_generator_resolves_pooled_and_legacy_speech_paths(tmp_path, monkeypatch):
    from pathlib import Path

    from code.generate_database import DataBaseGenerator

    monkeypatch.chdir(tmp_path)  # the generator creates ./cache
    gen = DataBaseGenerator([], [], [], [], [], [], [1000], 'octave band', 16000,
                            -60, 4, False, [0, 0], [0.2, 3.1, 0.1], [-6, 19, 1])
    split_dir = Path('data') / 'Speech' / 'train'

    # Pooled entries (speaker split) are relative to data/Speech, whatever the split.
    assert gen._speech_file_path(split_dir, 'test/F1s4.wav') == Path('data/Speech/test/F1s4.wav')
    # Bare names (v0 configs) are looked up in the split folder.
    assert gen._speech_file_path(split_dir, 'F1s3.wav') == Path('data/Speech/train/F1s3.wav')


def test_cache_name_depends_on_the_split(tmp_path, monkeypatch):
    """LORO folds differ only in their split; they must never share a cache."""
    from code.generate_database import DataBaseGenerator

    monkeypatch.chdir(tmp_path)

    def db_name(rirs_train, rirs_test):
        return DataBaseGenerator(['F1s3.wav'], ['M4s4.wav'], rirs_train + rirs_test, [],
                                 rirs_train, rirs_test, [1000], 'octave band', 16000,
                                 -60, 4, False, [0, 40], [0.2, 3.1, 0.1], [-6, 19, 1]).db_name

    fold_a = db_name(['great_hall_x00y00.wav'], ['classroom_00x00y.wav'])
    fold_b = db_name(['classroom_00x00y.wav'], ['great_hall_x00y00.wav'])
    assert fold_a != fold_b
    assert fold_a == db_name(['great_hall_x00y00.wav'], ['classroom_00x00y.wav'])


# --------------------------------------------------------------------------- #
# Augmentation only on training RIRs (DATA-11)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('test_room', ROOMS)
def test_v1_configs_augment_training_rirs_only(monkeypatch, test_room):
    import configs.v1_base as v1_base

    monkeypatch.setattr(v1_base, 'list_rir_files', lambda _: sorted(ALL_RIRS))
    monkeypatch.setattr(v1_base, 'list_speech_files',
                        lambda _: SPEECH + ['test/F4s1.wav', 'test/F4s2.wav'])
    config = v1_base.make_config(exp_num=999, add_noise=False, test_rooms=(test_room,))

    augmented = config['to_augmentate']
    assert set(augmented) <= set(config['rirs_for_training'])
    assert {rir_group(f) for f in augmented} == set(ROOMS) - {test_room}
    # The training rooms keep exactly the selection they had when every room was augmented.
    everything = select_rirs_to_augment(ALL_RIRS, per_room=15, seed=v1_base.SEED)
    assert augmented == [f for f in everything if rir_group(f) != test_room]


def test_generator_never_augments_test_rirs(tmp_path, monkeypatch):
    from collections import Counter

    from code.generate_database import DataBaseGenerator

    monkeypatch.chdir(tmp_path)
    train, test = ['great_hall_x00y00.wav'], ['classroom_00x00y.wav']
    gen = DataBaseGenerator([], [], train + test, train + test, train, test, [1000],
                            'octave band', 16000, -60, 4, False, [0, 40], [0.5, 1.1, 0.5],
                            [0, 7, 6], seed=SEED)
    assert gen._should_augment('great_hall_x00y00')
    # Listed in to_augmentate, but a test RIR: its variants would end up in the test set.
    assert not gen._should_augment('classroom_00x00y')
    rir = np.zeros(8000)
    rir[100] = 1.0
    assert gen._augmented_variants('classroom_00x00y', rir, Counter()) == []
