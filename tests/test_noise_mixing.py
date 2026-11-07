"""Tests for broadband noise mixing before the filter bank (BUG-03)."""

import numpy as np
import pytest

from code.generate_database import DataBaseGenerator
from code.mixing import mix_at_snr
from code.parameters_calculation import pink_noise

FS = 16000
BANDS = [125, 1000, 4000]
SNR_RANGE = [-5, 20]


def make_generator(add_noise: bool) -> DataBaseGenerator:
    return DataBaseGenerator(
        [], [], [], [], [], [], BANDS, 'octave band', FS,
        -60, 4, add_noise, SNR_RANGE, [0.2, 3.1, 0.1], [-6, 19, 1],
    )


@pytest.fixture
def in_tmp_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the generator creates ./cache


@pytest.fixture
def reverberant_speech():
    rng = np.random.default_rng(7)
    # Speech-like stand-in: low-pass-ish noise with an on/off envelope.
    x = np.cumsum(rng.standard_normal(5 * FS))
    x -= np.convolve(x, np.ones(64) / 64, mode="same")
    x *= (np.sin(2 * np.pi * 3 * np.arange(x.size) / FS) > 0)
    return x / np.max(np.abs(x))


# --------------------------------------------------------------------------- #
# mix_at_snr
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("snr_db", [-5.0, 0.0, 7.3, 20.0])
def test_mix_reaches_target_snr(snr_db):
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(10_000)
    noise = 3.7 * rng.standard_normal(10_000)

    mixture, gain = mix_at_snr(signal, noise, snr_db)

    measured = 10 * np.log10(np.mean(signal**2) / np.mean((gain * noise) ** 2))
    assert measured == pytest.approx(snr_db, abs=1e-9)
    np.testing.assert_allclose(mixture, signal + gain * noise)


def test_mix_does_not_depend_on_noise_level():
    rng = np.random.default_rng(1)
    signal, noise = rng.standard_normal(1000), rng.standard_normal(1000)
    mix_a, _ = mix_at_snr(signal, noise, 10.0)
    mix_b, _ = mix_at_snr(signal, 100 * noise, 10.0)
    np.testing.assert_allclose(mix_a, mix_b)


@pytest.mark.parametrize(
    "signal, noise, snr_db",
    [
        (np.ones((4, 2)), np.ones((4, 2)), 0.0),
        (np.ones(4), np.ones(5), 0.0),
        (np.array([]), np.array([]), 0.0),
        (np.zeros(4), np.ones(4), 0.0),
        (np.ones(4), np.zeros(4), 0.0),
        (np.ones(4), np.ones(4), np.inf),
    ],
)
def test_mix_invalid_inputs_raise(signal, noise, snr_db):
    with pytest.raises(ValueError):
        mix_at_snr(signal, noise, snr_db)


def test_pink_noise_is_reproducible_with_rng():
    a = pink_noise(4096, rng=np.random.default_rng(3))
    b = pink_noise(4096, rng=np.random.default_rng(3))
    np.testing.assert_array_equal(a, b)
    assert np.max(np.abs(a)) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# DataBaseGenerator._build_input_bands
# --------------------------------------------------------------------------- #

def test_noise_is_added_before_the_filter_bank(in_tmp_dir, reverberant_speech):
    generator = make_generator(add_noise=True)

    input_bands, snr_db, band_snr_db = generator._build_input_bands(
        reverberant_speech, rng=np.random.default_rng(11)
    )

    # Rebuild the expected recording with the same random draws: mix, then filter.
    rng = np.random.default_rng(11)
    expected_snr = rng.uniform(*SNR_RANGE)
    noise = pink_noise(reverberant_speech.size, rng=rng)
    mixture, gain = mix_at_snr(reverberant_speech, noise, expected_snr)
    expected = generator.bp_filter.filter_signals(mixture.astype(np.float32))

    assert snr_db == pytest.approx(expected_snr)
    assert SNR_RANGE[0] <= snr_db <= SNR_RANGE[1]
    np.testing.assert_allclose(input_bands, expected, atol=1e-9)

    # Diagnostic per-band SNR matches filtering speech and noise separately.
    speech_bands = generator.bp_filter.filter_signals(reverberant_speech.astype(np.float32))
    noise_bands = generator.bp_filter.filter_signals((gain * noise).astype(np.float32))
    expected_band_snr = 10 * np.log10(
        np.mean(speech_bands**2, axis=1) / np.mean(noise_bands**2, axis=1)
    )
    np.testing.assert_allclose(band_snr_db, expected_band_snr, atol=1e-3)


def test_low_band_contains_no_out_of_band_noise(in_tmp_dir, reverberant_speech):
    """
    Regression test for BUG-03: the 125 Hz band used to receive unfiltered broadband
    noise. Now its spectrum above 2 kHz must be far below its in-band level.
    """
    generator = make_generator(add_noise=True)
    input_bands, _, _ = generator._build_input_bands(
        reverberant_speech, rng=np.random.default_rng(5)
    )

    spectrum = np.abs(np.fft.rfft(input_bands[0])) ** 2
    freqs = np.fft.rfftfreq(input_bands.shape[1], 1 / FS)
    in_band = spectrum[(freqs > 88) & (freqs < 177)].mean()
    out_of_band = spectrum[freqs > 2000].mean()
    assert 10 * np.log10(out_of_band / in_band) < -60


def test_without_noise_bands_are_the_filtered_signal(in_tmp_dir, reverberant_speech):
    generator = make_generator(add_noise=False)

    input_bands, snr_db, band_snr_db = generator._build_input_bands(reverberant_speech)

    expected = generator.bp_filter.filter_signals(reverberant_speech.astype(np.float32))
    np.testing.assert_allclose(input_bands, expected)
    assert np.isnan(snr_db)
    assert np.all(np.isnan(band_snr_db))


def test_entry_records_share_one_broadband_snr(in_tmp_dir, reverberant_speech):
    generator = make_generator(add_noise=True)
    rng = np.random.default_rng(9)
    rir = rng.standard_normal(FS) * np.exp(-6.9 * np.arange(FS) / FS / 0.5)

    records = generator._process_entry(
        reverberant_speech, rir, 'speech', 'rir', 'train', 'original'
    )

    assert {r['band'] for r in records} == set(BANDS)
    assert len({r['snr'] for r in records}) == 1
    assert all(np.isfinite(r['snr_band']) for r in records)
    assert all(len(r['tae']) == 200 for r in records)
