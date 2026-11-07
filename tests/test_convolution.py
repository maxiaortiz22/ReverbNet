"""Tests for the causal convolution used to build reverberant speech (BUG-01)."""

import numpy as np
import pytest
from scipy.signal import fftconvolve

from code.convolution import causal_convolve


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


def test_unit_impulse_returns_the_rir(rng):
    rir = rng.standard_normal(500)
    signal = np.zeros(2000)
    signal[0] = 1.0

    out = causal_convolve(signal, rir)

    np.testing.assert_allclose(out[: rir.size], rir, atol=1e-12)
    np.testing.assert_allclose(out[rir.size:], 0.0, atol=1e-12)


def test_delayed_impulse_shifts_the_rir(rng):
    rir = rng.standard_normal(300)
    delay = 123
    signal = np.zeros(1000)
    signal[delay] = 1.0

    out = causal_convolve(signal, rir)

    np.testing.assert_allclose(out[:delay], 0.0, atol=1e-12)
    np.testing.assert_allclose(out[delay:delay + rir.size], rir, atol=1e-12)


@pytest.mark.parametrize("n_signal, n_rir", [(1000, 10), (1000, 1000), (1000, 5000)])
def test_output_length_matches_signal(rng, n_signal, n_rir):
    out = causal_convolve(rng.standard_normal(n_signal), rng.standard_normal(n_rir))
    assert out.shape == (n_signal,)


def test_matches_direct_convolution(rng):
    signal = rng.standard_normal(800)
    rir = rng.standard_normal(150)

    expected = np.convolve(signal, rir)[: signal.size]

    np.testing.assert_allclose(causal_convolve(signal, rir), expected, atol=1e-9)


def test_alignment_does_not_depend_on_rir_length(rng):
    """
    Regression test for BUG-01.

    Two RIRs that share their first taps but differ in total length (as synthetic
    RIRs of different reverberation times do) must produce the same output at the
    start of the window. With ``mode='same'`` the output is shifted by
    ``(len(rir) - 1) // 2``, so the RIR length leaks into the features.
    """
    signal = rng.standard_normal(4000)
    head = rng.standard_normal(200)
    short_rir = np.concatenate([head, np.zeros(300)])
    long_rir = np.concatenate([head, np.zeros(2800)])

    out_short = causal_convolve(signal, short_rir)
    out_long = causal_convolve(signal, long_rir)
    np.testing.assert_allclose(out_short, out_long, atol=1e-9)

    # The previous implementation fails this property.
    same_short = fftconvolve(signal, short_rir, mode="same")
    same_long = fftconvolve(signal, long_rir, mode="same")
    assert not np.allclose(same_short, same_long, atol=1e-6)


@pytest.mark.parametrize(
    "signal, rir",
    [
        (np.zeros((10, 2)), np.ones(3)),
        (np.zeros(10), np.ones((3, 1))),
        (np.array([]), np.ones(3)),
        (np.zeros(10), np.array([])),
    ],
)
def test_invalid_inputs_raise(signal, rir):
    with pytest.raises(ValueError):
        causal_convolve(signal, rir)
