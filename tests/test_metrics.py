"""Tests for code/metrics.py (TRAIN-08): record metadata and error metrics."""

import numpy as np
import pandas as pd
import pytest

from code.metrics import (DESCRIPTORS, build_test_metadata, denormalize, descriptor_metrics,
                          metrics_by_group, record_metadata)


def test_record_metadata_parses_every_rir_type():
    meta = record_metadata([
        'F4s1|classroom_00x05y|original',
        'M9s2|octagon_x12y11|TR_var_0.80',
        'M4s3|great_hall_x05y02|TR_1.10_DRR_var_3.00',
        'F4s2|sintetica_Seed1000169287_Tr1.5|original',
    ])
    assert meta['speaker'].tolist() == ['F4', 'M9', 'M4', 'F4']
    assert meta['room'].tolist() == ['classroom', 'octagon', 'great_hall', 'synthetic']
    assert meta['rir_type'].tolist() == ['real', 'augmented', 'augmented', 'synthetic']
    assert meta['augmentation'].tolist() == [
        'original', 'TR_var_0.80', 'TR_1.10_DRR_var_3.00', 'original']
    assert meta['rir'].iloc[0] == 'classroom_00x05y'


@pytest.mark.parametrize('name', [
    'F4s1|classroom_00x05y', 'F4s1|unknown_room|original', 'X1|classroom_a|original'])
def test_record_metadata_rejects_malformed_names(name):
    with pytest.raises(ValueError):
        record_metadata([name])


def test_build_test_metadata_keeps_order_and_extra_columns():
    records = pd.DataFrame({
        'ReverbedAudio': ['M9s2|octagon_x12y11|original', 'F4s1|classroom_00x05y|original'],
        'drr': [3.0, -1.5],
        'snr': [np.nan, 12.0],
    }, index=[10, 3])  # shuffled index, as left by DataFrame.sample
    meta = build_test_metadata(records)
    assert meta.index.tolist() == [0, 1]
    assert meta['room'].tolist() == ['octagon', 'classroom']
    assert meta['drr'].tolist() == [3.0, -1.5]
    assert 'snr_band' not in meta.columns


def test_descriptor_metrics_known_values():
    true = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([1.5, 2.0, 2.0, 4.0])  # errors +0.5, 0, -1, 0
    m = descriptor_metrics(pred, true, 'C50')
    assert m['n'] == 4
    assert m['mae'] == pytest.approx(0.375)
    assert m['bias'] == pytest.approx(-0.125)
    assert m['rmse'] == pytest.approx(np.sqrt(1.25 / 4))
    assert m['within_jnd'] == pytest.approx(1.0)  # |err| <= 1 dB for all four


def test_descriptor_metrics_t30_jnd_is_relative():
    true = np.array([1.0, 2.0])
    pred = np.array([1.04, 2.2])  # 4 % and 10 % errors
    assert descriptor_metrics(pred, true, 'T30')['within_jnd'] == pytest.approx(0.5)


def test_descriptor_metrics_perfect_and_degenerate():
    x = np.linspace(0, 1, 10)
    m = descriptor_metrics(x, x, 'D50')
    assert m['mae'] == 0 and m['spearman'] == pytest.approx(1) and m['pearson'] == pytest.approx(1)
    constant = descriptor_metrics(np.ones(5), np.arange(5.0), 'C80')
    assert np.isnan(constant['spearman']) and np.isnan(constant['pearson'])


def test_descriptor_metrics_validates_inputs():
    with pytest.raises(ValueError):
        descriptor_metrics(np.ones(3), np.ones(4), 'C50')
    with pytest.raises(ValueError):
        descriptor_metrics(np.ones(0), np.ones(0), 'C50')
    with pytest.raises(ValueError):
        descriptor_metrics(np.ones(3), np.ones(3), 'EDT')


def test_denormalize_accepts_model_shapes():
    norm = [2.0, 10.0, 10.0, 100.0]
    y = np.ones((3, 4, 1)) * 0.5
    assert np.allclose(denormalize(y, norm), [[1.0, 5.0, 5.0, 50.0]] * 3)


def test_metrics_by_group_splits_records():
    true = np.zeros((4, 4))
    pred = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [-2, -2, -2, -2], [-2, -2, -2, -2]], dtype=float)
    table = metrics_by_group(pred, true, ['real', 'real', 'synthetic', 'synthetic'])
    assert set(table['group']) == {'real', 'synthetic'}
    assert len(table) == 2 * len(DESCRIPTORS)
    bias = table.set_index(['group', 'descriptor'])['bias']
    assert bias['real', 'C50'] == pytest.approx(1.0)
    assert bias['synthetic', 'D50'] == pytest.approx(-2.0)
    overall = metrics_by_group(pred, true)
    assert overall['group'].unique().tolist() == ['all']
    with pytest.raises(ValueError):
        metrics_by_group(pred, true, ['a', 'b'])
