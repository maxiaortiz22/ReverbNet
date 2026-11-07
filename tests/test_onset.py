"""Tests for direct-sound onset detection and onset-referenced descriptors (BUG-02)."""

import numpy as np
import pytest

from code.cpp import audio_processing
from code.generate_database import DataBaseGenerator
from code.parameters_calculation import find_onset, trim_to_onset

FS = 16000


def ideal_decay(rt: float, duration: float, fs: int = FS) -> np.ndarray:
    """Amplitude envelope whose energy decays exactly 60 dB in ``rt`` seconds."""
    t = np.arange(int(duration * fs)) / fs
    a = 6 * np.log(10) / rt  # energy decay constant: h^2 = exp(-a t)
    return np.exp(-a * t / 2)


def analytic_clarity(rt: float, te: float) -> float:
    a = 6 * np.log(10) / rt
    return 10 * np.log10(np.exp(a * te) - 1)


def analytic_definition(rt: float) -> float:
    a = 6 * np.log(10) / rt
    return 100 * (1 - np.exp(-a * 0.05))


# --------------------------------------------------------------------------- #
# find_onset / trim_to_onset
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("position", [0, 1, 250, 799])
def test_impulse_onset_is_its_position(position):
    rir = np.zeros(800)
    rir[position] = 1.0
    assert find_onset(rir) == position


def test_negative_polarity_direct_sound():
    rir = np.zeros(1000)
    rir[300] = -1.0
    rir[400:] = 0.05
    assert find_onset(rir) == 300


def test_low_level_pre_delay_noise_is_ignored():
    rng = np.random.default_rng(0)
    delay = 528  # 33 ms at 16 kHz, like the Great Hall RIRs
    decay = ideal_decay(rt=1.0, duration=1.0)
    rir = np.concatenate([np.zeros(delay), decay])
    rir += 1e-3 * rng.standard_normal(rir.size)  # -60 dB background noise
    assert find_onset(rir) == delay


def test_onset_never_after_peak():
    rng = np.random.default_rng(1)
    for _ in range(20):
        rir = rng.standard_normal(2000) * ideal_decay(rt=0.5, duration=2000 / FS)
        assert find_onset(rir) <= np.argmax(np.abs(rir))


def test_threshold_controls_detection():
    rir = np.array([0.0, 0.05, 0.2, 1.0, 0.5])
    assert find_onset(rir, threshold_db=-20.0) == 2  # 0.2 >= 0.1
    assert find_onset(rir, threshold_db=-30.0) == 1  # 0.05 >= 0.0316


def test_trim_to_onset():
    rir = np.array([0.0, 0.0, 1.0, 0.5, 0.25])
    np.testing.assert_array_equal(trim_to_onset(rir), [1.0, 0.5, 0.25])


@pytest.mark.parametrize(
    "rir, threshold_db",
    [
        (np.zeros((10, 2)), -20.0),
        (np.array([]), -20.0),
        (np.zeros(10), -20.0),
        (np.ones(10), 0.0),
    ],
)
def test_invalid_inputs_raise(rir, threshold_db):
    with pytest.raises(ValueError):
        find_onset(rir, threshold_db)


# --------------------------------------------------------------------------- #
# Descriptors referenced to the onset
# --------------------------------------------------------------------------- #

def test_onset_referenced_descriptors_match_analytic_values():
    """
    With a 33 ms pre-delay, integrating from the onset recovers the analytic values of
    an ideal exponential decay (T = 1 s), while integrating from sample 0 (the thesis
    behavior) is off by several dB.
    """
    rt = 1.0
    rir = np.concatenate([np.zeros(528), ideal_decay(rt, duration=3.0)])
    trimmed = trim_to_onset(rir)

    c50 = audio_processing.ClarityCalculator.calculate(50, trimmed, FS)
    c80 = audio_processing.ClarityCalculator.calculate(80, trimmed, FS)
    d50 = audio_processing.DefinitionCalculator.calculate(trimmed, FS)

    assert c50 == pytest.approx(analytic_clarity(rt, 0.05), abs=0.05)  # ≈ -0.02 dB
    assert c80 == pytest.approx(analytic_clarity(rt, 0.08), abs=0.05)  # ≈ 3.05 dB
    assert d50 == pytest.approx(analytic_definition(rt), abs=0.2)      # ≈ 49.9 %

    c50_from_zero = audio_processing.ClarityCalculator.calculate(50, rir, FS)
    assert c50 - c50_from_zero > 3.0


def test_generator_descriptors_are_invariant_to_pre_delay(tmp_path, monkeypatch):
    """C50/C80/D50 produced by the database generator must not depend on the pre-delay."""
    monkeypatch.chdir(tmp_path)  # the generator creates ./cache
    rng = np.random.default_rng(42)
    bands = [500, 1000, 4000]
    generator = DataBaseGenerator(
        [], [], [], [], [], [], bands, 'octave band', FS,
        -60, 4, False, [-5, 20], [0.2, 3.1, 0.1], [-6, 19, 1],
    )

    speech = rng.standard_normal(5 * FS)
    rir = rng.standard_normal(2 * FS) * ideal_decay(rt=1.0, duration=2.0)
    rir /= np.max(np.abs(rir))
    delayed_rir = np.concatenate([np.zeros(528), rir])

    def descriptors(rir_data):
        records = generator._process_entry(speech, rir_data, 'speech', 'rir', 'train', 'original')
        return {r['band']: np.array(r['descriptors']) for r in records}

    plain = descriptors(rir)
    delayed = descriptors(delayed_rir)

    assert set(plain) == set(delayed) == set(bands)
    for band in bands:
        # [T30, C50, C80, D50]: C50/C80 in dB, D50 in %.
        np.testing.assert_allclose(delayed[band][1:3], plain[band][1:3], atol=0.05)
        np.testing.assert_allclose(delayed[band][3], plain[band][3], atol=0.2)
