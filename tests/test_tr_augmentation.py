"""Regression tests for the TR augmentation decay analysis (BUG-09)."""

import importlib

import numpy as np
import pytest

from code.parameters_calculation import NoiseError, TrAugmentationError, tr_lundeby
from filters import BandpassFilter

# The package re-exports the *function* tr_augmentation under the module's name, so the
# module itself has to be imported explicitly.
tra = importlib.import_module('code.parameters_calculation.tr_augmentation')

FS = 16000


def decay_with_noise_floor(rt=1.0, noise_db=-50.0, duration=2.0, seed=0):
    """Exponential decay (amplitude falls 60 dB in ``rt``) on top of a constant noise floor."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * FS)) / FS
    decay = rng.standard_normal(t.size) * np.exp(-6.908 * t / rt)
    floor = 10 ** (noise_db / 20) * rng.standard_normal(t.size)
    rir = decay + floor
    return rir / np.max(np.abs(rir))


# --------------------------------------------------------------------------- #
# get_valid_interval
# --------------------------------------------------------------------------- #

def test_valid_interval_runs_to_the_end_when_the_floor_is_never_crossed():
    arr = np.array([0.0, -1.0, -2.0, -3.0, -4.0])
    values, samples = tra.get_valid_interval(arr, init_value=-0.5, final_value=-10.0)
    np.testing.assert_array_equal(values, [-1.0, -2.0, -3.0, -4.0])
    np.testing.assert_array_equal(samples, [1, 2, 3, 4])


def test_valid_interval_stops_at_the_lower_threshold():
    arr = np.array([0.0, -1.0, -2.0, -3.0, -4.0])
    values, samples = tra.get_valid_interval(arr, init_value=-0.5, final_value=-2.5)
    np.testing.assert_array_equal(values, [-1.0, -2.0])
    np.testing.assert_array_equal(samples, [1, 2])


@pytest.mark.parametrize("init_value, final_value", [(-10.0, -20.0), (-0.5, -1.5)])
def test_valid_interval_raises_value_error_when_empty_or_too_short(init_value, final_value):
    with pytest.raises(ValueError):
        tra.get_valid_interval(np.array([0.0, -1.0, -2.0]), init_value, final_value)


# --------------------------------------------------------------------------- #
# Lundeby_method
# --------------------------------------------------------------------------- #

def test_lundeby_method_finds_the_decay_noise_crossing():
    """
    Regression test for BUG-09. On this band-filtered decay (RT 2 s, 2 kHz band, floor at
    −60 dB) the original code searched the late-decay fit *below* the noise floor and raised
    IndexError. The cross point must now be found where the decay meets the floor (the
    amplitude falls 30 dB/s, so roughly 1.5–2 s).
    """
    bank = BandpassFilter('octave band', FS, 4, [2000])
    rir = bank.filter_signals(decay_with_noise_floor(rt=2.0, noise_db=-60.0, seed=0).astype(np.float32))[0]
    cross_s = tra.Lundeby_method(rir, FS) / FS
    assert 1.3 < cross_s < 2.2


def test_lundeby_method_cross_point_on_a_broadband_decay():
    """Broadband decay on a −50 dB floor (60 dB/s amplitude decay → crossing near 0.83 s)."""
    cross_s = tra.Lundeby_method(decay_with_noise_floor(rt=1.0, noise_db=-50.0), FS) / FS
    assert 0.6 < cross_s < 1.1


def test_non_decaying_fit_raises_value_error():
    with pytest.raises(ValueError, match="not negative"):
        tra._decaying_regression(np.arange(10), np.linspace(-5, 0, 10))


# --------------------------------------------------------------------------- #
# tr_augmentation output checks
# --------------------------------------------------------------------------- #

def test_non_finite_augmentation_is_a_counted_failure(monkeypatch):
    bank = BandpassFilter('octave band', FS, 4, [500, 1000])
    monkeypatch.setattr(tra, 'augmentation', lambda rir, *args: np.full_like(rir, np.inf))

    with pytest.raises(TrAugmentationError) as info:
        tra.tr_augmentation(decay_with_noise_floor(noise_db=-70.0), FS, 2.0, bank,
                            rng=np.random.default_rng(0))
    assert info.value.cause == 'FloatingPointError'


def test_augmentation_of_a_decay_on_a_noise_floor_succeeds():
    bank = BandpassFilter('octave band', FS, 4, [500, 1000, 2000])
    rir = decay_with_noise_floor(rt=0.8, noise_db=-55.0)
    out = tra.tr_augmentation(rir, FS, 1.2, bank, rng=np.random.default_rng(0))
    assert out.shape == rir.shape
    assert np.all(np.isfinite(out))


# --------------------------------------------------------------------------- #
# tr_lundeby input validation and error types
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("rir", [np.array([]), np.zeros(1000), np.array([1.0, np.nan, 0.5])])
def test_tr_lundeby_rejects_invalid_input(rir):
    with pytest.raises(ValueError):
        tr_lundeby(rir, FS, -45)


def test_missing_dynamic_range_is_a_noise_error():
    """An impulse on a −50 dB floor has no decay interval 10 dB above the noise."""
    rng = np.random.default_rng(0)
    rir = 10 ** (-50 / 20) * rng.standard_normal(FS)
    rir[0] = 1.0
    with pytest.raises(NoiseError):
        tr_lundeby(rir, FS, -45)
