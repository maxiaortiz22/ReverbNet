"""Tests for batched prediction, errors and result saving in code/model.py (TRAIN-07/08)."""

import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from code.model import descriptors_err, prediction, save_exp_data


class FakeModel:
    """Keras-like model: output = mean of the input frames, repeated 4 times, plus offsets."""

    def __init__(self):
        self.calls = []

    def predict(self, X, batch_size=32, verbose=1):
        self.calls.append((len(X), batch_size, verbose))
        mean = X.mean(axis=(1, 2))
        return np.stack([mean, mean + 0.001, mean + 0.0123, mean - 0.456], axis=1)

    def save_weights(self, path):
        open(path, 'wb').close()


def test_prediction_is_one_batched_unrounded_call():
    rng = np.random.default_rng(0)
    X = rng.random((50, 200, 1))
    fake = FakeModel()
    pred = prediction(fake, X, batch_size=16)
    assert fake.calls == [(50, 16, 0)]
    assert pred.shape == (50, 4)
    # Not rounded to 2 decimals (the v0 behaviour).
    assert np.allclose(pred[:, 2] - pred[:, 0], 0.0123)


def test_prediction_validates_shapes():
    with pytest.raises(ValueError):
        prediction(FakeModel(), np.ones((5, 200)))


def test_descriptors_err_is_vectorized_and_unrounded():
    pred = np.array([[0.123, 0.5, 0.25, 1.0], [0.0, 0.0, 0.0, 0.0]])
    y = np.array([[0.1, 0.5, 0.2501, 0.9], [0.333, 0.0, 0.0, 0.0]]).reshape(2, 4, 1)
    err_t30, err_c50, err_c80, err_d50 = descriptors_err(pred, y)
    assert np.allclose(err_t30, [0.023, -0.333])
    assert np.allclose(err_c80, [-0.0001, 0.0])
    with pytest.raises(ValueError):
        descriptors_err(pred, y[:1])


def test_save_exp_data_stores_metadata(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    history = SimpleNamespace(history={'loss': [3.0, 2.0, 1.5], 'val_loss': [3.0, 1.0, 2.0]})
    y = np.zeros((2, 4, 1))
    pred = np.ones((2, 4))
    meta = pd.DataFrame({'recording': ['a|b|c', 'd|e|f']}, index=[7, 3])
    save_exp_data(999, 500, FakeModel(), history, pred, *descriptors_err(pred, y),
                  1.0, 1.0, 1.0, 1.0, np.zeros((2, 200, 1)), y, test_metadata=meta)
    with open(tmp_path / 'results' / 'exp999' / 'results_500.pickle', 'rb') as handle:
        res = pickle.load(handle)
    assert res['best_epoch'] == 2
    assert res['test_metadata']['recording'].tolist() == ['a|b|c', 'd|e|f']
    assert res['test_metadata'].index.tolist() == [0, 1]
    with pytest.raises(ValueError):
        save_exp_data(999, 250, FakeModel(), history, pred, *descriptors_err(pred, y),
                      1.0, 1.0, 1.0, 1.0, np.zeros((2, 200, 1)), y, test_metadata=meta.iloc[:1])
