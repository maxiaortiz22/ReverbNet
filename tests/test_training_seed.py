"""Tests for the per-band training seeds (TRAIN-09)."""

import sys
from types import SimpleNamespace

import pytest

from code.model import seed_training
from code.random_streams import RNG_STREAMS, training_seed

SEED = 2222


def test_training_seed_is_stable_and_in_range():
    value = training_seed(SEED, 1000)
    assert value == training_seed(SEED, 1000)
    assert isinstance(value, int)
    assert 0 <= value < 2**31


def test_training_seed_depends_on_seed_band_and_replica():
    values = {
        training_seed(SEED, 1000),
        training_seed(SEED + 1, 1000),
        training_seed(SEED, 2000),
        training_seed(SEED, 1000, replica=1),
        training_seed(SEED, 1000, replica=2),
    }
    assert len(values) == 5


def test_every_band_gets_its_own_seed():
    bands = [125, 250, 500, 1000, 2000, 4000, 8000]
    assert len({training_seed(SEED, band) for band in bands}) == len(bands)


def test_training_seed_rejects_negative_replica():
    with pytest.raises(ValueError):
        training_seed(SEED, 1000, replica=-1)


def test_existing_stream_ids_are_unchanged():
    # Renumbering a stream would silently change reproducible databases and splits.
    assert {k: RNG_STREAMS[k] for k in ('noise', 'split', 'validation', 'augment',
                                        'drr_subset', 'tr_tail')} == {
        'noise': 1, 'split': 2, 'validation': 3, 'augment': 4, 'drr_subset': 5, 'tr_tail': 6}
    assert RNG_STREAMS['training'] == 7


def test_seed_training_calls_keras(monkeypatch):
    calls = []
    fake_tf = SimpleNamespace(keras=SimpleNamespace(
        utils=SimpleNamespace(set_random_seed=calls.append)))
    monkeypatch.setitem(sys.modules, 'tensorflow', fake_tf)
    seed_training(1234)
    assert calls == [1234]
