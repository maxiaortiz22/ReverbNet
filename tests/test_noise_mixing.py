"""
Tests for noise mixing (BUG-03) and reproducible noise settings (DATA-09).

Noise is mixed broadband before the filter bank; its color, SNR and the clean/noisy
decision are drawn per recording from a generator seeded with the project seed and
the recording name.
"""

import numpy as np
import pytest
from scipy.signal import welch

from code.generate_database import RNG_STREAMS, DataBaseGenerator
from code.mixing import mix_at_snr
from code.parameters_calculation import colored_noise, pink_noise

FS = 16000
BANDS = [125, 1000, 4000]
SNR_RANGE = [0, 40]
BETA_RANGE = [1.0, 2.0]


def make_generator(add_noise=True, seed=2222, clean_fraction=0.2, noise_beta=BETA_RANGE,
                   snr=SNR_RANGE) -> DataBaseGenerator:
    return DataBaseGenerator(
        [], [], [], [], [], [], BANDS, 'octave band', FS,
        -60, 4, add_noise, snr, [0.2, 3.1, 0.1], [-6, 19, 1],
        noise_beta=noise_beta, clean_fraction=clean_fraction, seed=seed,
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


@pytest.fixture
def rir():
    rng = np.random.default_rng(9)
    return rng.standard_normal(FS) * np.exp(-6.9 * np.arange(FS) / FS / 0.5)


# --------------------------------------------------------------------------- #
# mix_at_snr and colored noise
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


def test_noise_is_reproducible_with_rng():
    a = pink_noise(4096, rng=np.random.default_rng(3))
    b = pink_noise(4096, rng=np.random.default_rng(3))
    np.testing.assert_array_equal(a, b)
    assert np.max(np.abs(a)) == pytest.approx(1.0)


@pytest.mark.parametrize("beta", [0.0, 1.0, 1.5, 2.0])
def test_colored_noise_has_the_requested_spectral_slope(beta):
    noise = colored_noise(2**18, beta, rng=np.random.default_rng(4))
    freqs, psd = welch(noise, fs=FS, nperseg=4096)
    band = (freqs >= 100) & (freqs <= 4000)
    slope = np.polyfit(np.log10(freqs[band]), np.log10(psd[band]), 1)[0]
    assert slope == pytest.approx(-beta, abs=0.15)


# --------------------------------------------------------------------------- #
# Reproducible per-recording noise settings
# --------------------------------------------------------------------------- #

def test_noise_stream_ids_are_unique():
    assert len(set(RNG_STREAMS.values())) == len(RNG_STREAMS)


def test_settings_depend_only_on_seed_and_recording(in_tmp_dir):
    gen = make_generator()
    names = [f'F{i}s3|great_hall_x{i:02d}|original' for i in range(50)]

    first = [gen._draw_noise_settings(gen._noise_rng(n)) for n in names]
    # Reverse order and a fresh generator with the same config: identical draws.
    fresh = make_generator()
    again = [fresh._draw_noise_settings(fresh._noise_rng(n)) for n in reversed(names)]
    assert first == list(reversed(again))

    other_seed = make_generator(seed=1)
    assert first != [other_seed._draw_noise_settings(other_seed._noise_rng(n)) for n in names]
    assert len({s for s in first if s is not None}) == sum(s is not None for s in first)


def test_settings_follow_the_configured_distribution(in_tmp_dir):
    gen = make_generator(clean_fraction=0.2)
    settings = [gen._draw_noise_settings(gen._noise_rng(f'speech{i}|rir|original'))
                for i in range(4000)]

    noisy = np.array([s for s in settings if s is not None])
    assert 1 - len(noisy) / len(settings) == pytest.approx(0.2, abs=0.03)
    assert noisy[:, 0].min() >= BETA_RANGE[0] and noisy[:, 0].max() <= BETA_RANGE[1]
    assert noisy[:, 1].min() >= SNR_RANGE[0] and noisy[:, 1].max() <= SNR_RANGE[1]
    assert noisy[:, 0].mean() == pytest.approx(1.5, abs=0.05)
    assert noisy[:, 1].mean() == pytest.approx(20.0, abs=1.0)


def test_entry_records_are_reproducible(in_tmp_dir, reverberant_speech, rir):
    def records(gen):
        return gen._process_entry(reverberant_speech, rir, 'M1s3', 'octagon_x01', 'train', 'original')

    a = records(make_generator(clean_fraction=0.0))
    b = records(make_generator(clean_fraction=0.0))
    c = records(make_generator(clean_fraction=0.0, seed=7))

    assert [r['band'] for r in a] == BANDS
    for ra, rb in zip(a, b):
        np.testing.assert_array_equal(ra['tae'], rb['tae'])
        assert (ra['snr'], ra['noise_beta']) == (rb['snr'], rb['noise_beta'])
    assert a[0]['snr'] != c[0]['snr']


# --------------------------------------------------------------------------- #
# DataBaseGenerator._build_input_bands
# --------------------------------------------------------------------------- #

def test_noise_is_added_before_the_filter_bank(in_tmp_dir, reverberant_speech):
    gen = make_generator(clean_fraction=0.0)

    input_bands, snr_db, band_snr_db, beta = gen._build_input_bands(
        reverberant_speech, np.random.default_rng(11)
    )

    # Rebuild the expected recording with the same draws, in the documented order.
    rng = np.random.default_rng(11)
    assert rng.random() >= 0.0  # clean/noisy decision
    expected_beta = rng.uniform(*BETA_RANGE)
    expected_snr = rng.uniform(*SNR_RANGE)
    noise = colored_noise(reverberant_speech.size, expected_beta, rng=rng)
    mixture, gain = mix_at_snr(reverberant_speech, noise, expected_snr)
    expected = gen.bp_filter.filter_signals(mixture.astype(np.float32))

    assert (beta, snr_db) == pytest.approx((expected_beta, expected_snr))
    np.testing.assert_allclose(input_bands, expected, atol=1e-9)

    # Diagnostic per-band SNR matches filtering speech and noise separately.
    speech_bands = gen.bp_filter.filter_signals(reverberant_speech.astype(np.float32))
    noise_bands = gen.bp_filter.filter_signals((gain * noise).astype(np.float32))
    expected_band_snr = 10 * np.log10(
        np.mean(speech_bands**2, axis=1) / np.mean(noise_bands**2, axis=1)
    )
    np.testing.assert_allclose(band_snr_db, expected_band_snr, atol=1e-3)


def test_low_band_contains_no_out_of_band_noise(in_tmp_dir, reverberant_speech):
    """
    Regression test for BUG-03: the 125 Hz band used to receive unfiltered broadband
    noise. Now its spectrum above 2 kHz must be far below its in-band level.
    """
    gen = make_generator(clean_fraction=0.0, noise_beta=[1.0, 1.0], snr=[0, 0])
    input_bands, _, _, _ = gen._build_input_bands(reverberant_speech, np.random.default_rng(5))

    spectrum = np.abs(np.fft.rfft(input_bands[0])) ** 2
    freqs = np.fft.rfftfreq(input_bands.shape[1], 1 / FS)
    in_band = spectrum[(freqs > 88) & (freqs < 177)].mean()
    out_of_band = spectrum[freqs > 2000].mean()
    assert 10 * np.log10(out_of_band / in_band) < -60


@pytest.mark.parametrize("add_noise, clean_fraction", [(False, 0.0), (True, 1.0)])
def test_clean_recordings_are_the_filtered_signal(in_tmp_dir, reverberant_speech,
                                                  add_noise, clean_fraction):
    gen = make_generator(add_noise=add_noise, clean_fraction=clean_fraction)
    rng = np.random.default_rng(0) if add_noise else None

    input_bands, snr_db, band_snr_db, beta = gen._build_input_bands(reverberant_speech, rng)

    expected = gen.bp_filter.filter_signals(reverberant_speech.astype(np.float32))
    np.testing.assert_allclose(input_bands, expected)
    assert np.isnan(snr_db) and np.isnan(beta)
    assert np.all(np.isnan(band_snr_db))


def test_noisy_generation_requires_an_rng(in_tmp_dir, reverberant_speech):
    with pytest.raises(ValueError):
        make_generator()._build_input_bands(reverberant_speech, None)


def test_entry_records_share_one_recording_setting(in_tmp_dir, reverberant_speech, rir):
    gen = make_generator(clean_fraction=0.0)

    records = gen._process_entry(reverberant_speech, rir, 'speech', 'rir', 'train', 'original')

    assert {r['band'] for r in records} == set(BANDS)
    assert len({(r['snr'], r['noise_beta']) for r in records}) == 1
    assert all(np.isfinite(r['snr_band']) for r in records)
    assert all(len(r['tae']) == 200 for r in records)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"snr": [20, 0]},
        {"noise_beta": [2.0, 1.0]},
        {"clean_fraction": 1.5},
        {"clean_fraction": -0.1},
    ],
)
def test_invalid_noise_parameters_raise(in_tmp_dir, kwargs):
    with pytest.raises(ValueError):
        make_generator(**kwargs)


def test_cache_name_encodes_noise_configuration(in_tmp_dir):
    name = make_generator(seed=2222, clean_fraction=0.2).db_name
    assert 'beta_1.0_2.0' in name and 'clean_0.2' in name and 'seed_2222' in name
    assert make_generator(seed=1).db_name != name
