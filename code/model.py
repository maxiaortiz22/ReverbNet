"""
Model definitions and training utilities for blind acoustic descriptor estimation.

This module provides:
- A convenience factory for a 4‑layer 1D CNN (:func:`model`).
- A helper to create a Keras EarlyStopping callback (:func:`create_early_stopping_callback`).
- Data shaping / normalization utilities (:func:`reshape_data`, :func:`normalize_descriptors`).
- Simple batch‑wise prediction and error calculation helpers
  (:func:`prediction`, :func:`descriptors_err`).
- Experiment results persistence (:func:`save_exp_data`).

**Important:** Layer names (e.g., ``'Audio de entrada'``, ``'capa1_conv'``) are
used when saving/loading weights; do not change them unless you update all
dependent code and existing checkpoints.
"""

import numpy as np
import os
import pickle


def create_early_stopping_callback(patience=100, monitor='val_loss', restore_best_weights=True):
    """
    Create a configured Keras EarlyStopping callback.

    Parameters
    ----------
    patience : int, default=100
        Number of epochs with no improvement (as measured on ``monitor``)
        after which training will be stopped.
    monitor : str, default='val_loss'
        Quantity to monitor (e.g., ``'val_loss'``, ``'loss'``).
    restore_best_weights : bool, default=True
        If ``True``, restore model weights from the epoch with the best value
        of the monitored quantity.

    Returns
    -------
    tensorflow.keras.callbacks.EarlyStopping
        Configured callback instance.
    """
    import tensorflow as tf

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=patience,
        restore_best_weights=restore_best_weights,
        verbose=1
    )
    return early_stopping


def seed_training(seed: int) -> None:
    """
    Seed Python, NumPy and TensorFlow for one training run (TRAIN-09).

    Covers weight initialization, batch shuffling and dropout. Two processes with the same
    seed produced identical weights on CPU with this call alone (TF 2.16.1), so
    ``enable_op_determinism`` is not used. Determinism on other devices is not guaranteed.

    Parameters
    ----------
    seed : int
        Value from :func:`code.random_streams.training_seed`.
    """
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)


def model(filters: list = [32, 18, 8, 4],
          kernel_size: list = [10, 5, 5, 5],
          activation: list = ['relu','relu','relu','relu'],
          pool_size: list = [2,2,2],
          learning_rate: float = 0.001):
    """
    Build and compile the 4‑layer 1D CNN used for blind descriptor estimation.

    Default architecture (corresponds to Experiment configs):

    1. ``filters = [32, 18, 8, 4]``
    2. ``kernel_size = [10, 5, 5, 5]``
    3. ``activation = ['relu','relu','relu','relu']``
    4. ``pool_size = [2,2,2]``
    5. ``learning_rate = 0.001``

    The network expects inputs of shape ``(200, 1)`` (time‑aligned feature
    vectors / TAE samples) and outputs a 4‑value dense layer corresponding to
    the target acoustic descriptors (T30, C50, C80, D50) after any scaling
    applied upstream.

    Parameters
    ----------
    filters : list of int
        Convolutional filter counts per layer.
    kernel_size : list of int
        Convolutional kernel sizes per layer.
    activation : list of str
        Activation functions per conv layer.
    pool_size : list of int
        Max‑pool window sizes for the first three conv blocks.
    learning_rate : float
        Adam optimizer learning rate.

    Returns
    -------
    tensorflow.keras.Model
        Compiled Keras model ready for training.
    """
    import tensorflow as tf
    import tensorflow.keras.layers as tfkl

    tf.keras.backend.clear_session()

    # NOTE: Layer names intentionally left in Spanish to preserve checkpoint compatibility.
    audio_in = tfkl.Input((200,1), name='Audio de entrada')

    capa_1 = tfkl.Conv1D(filters=filters[0], kernel_size=(kernel_size[0]), activation=activation[0], name='capa1_conv')(audio_in)
    capa_1 = tfkl.MaxPool1D(pool_size=pool_size[0], name='capa1_pool')(capa_1)
    capa_1 = tfkl.BatchNormalization(name='Batch_capa1')(capa_1)

    capa_2 = tfkl.Conv1D(filters=filters[1], kernel_size=(kernel_size[1]), activation=activation[1], name='capa2_conv')(capa_1)
    capa_2 = tfkl.MaxPool1D(pool_size=pool_size[1], name='capa2_pool')(capa_2)
    capa_2 = tfkl.BatchNormalization(name='Batch_capa2')(capa_2)
    capa_2 = tfkl.Dropout(0.4, name='capa2_drop')(capa_2)

    capa_3 = tfkl.Conv1D(filters=filters[2], kernel_size=(kernel_size[2]), activation=activation[2], name='capa3_conv')(capa_2)
    capa_3 = tfkl.MaxPool1D(pool_size=pool_size[2], name='capa3_pool')(capa_3)
    capa_3 = tfkl.BatchNormalization(name='Batch_capa3')(capa_3)

    capa_4 = tfkl.Conv1D(filters=filters[3], kernel_size=(kernel_size[3]), activation=activation[3], name='capa4_conv')(capa_3)
    capa_4 = tfkl.Flatten()(capa_4)

    tr_pred = tfkl.Dense(4, name='Salida_prediccion')(capa_4)

    modelo = tf.keras.Model(inputs=[audio_in], outputs=[tr_pred])
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse'
    )
    return modelo


def reshape_data(tae, descriptors):
    """
    Reshape raw TAE and descriptor lists into arrays suitable for TensorFlow.

    Parameters
    ----------
    tae : sequence
        Iterable of TAE vectors (each 1D array‑like of length 200 expected).
    descriptors : sequence
        Iterable of descriptor vectors (each 4‑element array‑like).

    Returns
    -------
    X : ndarray, shape (n_samples, 200, 1)
        Model input array.
    y : ndarray, shape (n_samples, 4, 1)
        Model target array.
    """
    tae_list = [[]] * int(len(tae))          # Pre‑allocate list of empty lists.
    descriptors_list = [[]] * int(len(descriptors))  # Pre‑allocate list of empty lists.

    for i in range(len(tae)):
        tae_list[i] = np.array(tae[i]).reshape(-1, 1)
        descriptors_list[i] = np.array(descriptors[i]).reshape(-1, 1)

    X = np.array(tae_list)
    y = np.array(descriptors_list)
    return X, y


def normalize_descriptors(y_train, y_val, y_test):
    """
    Normalize descriptor targets by the 95th percentile of the **training** set.

    The percentiles are fitted on ``y_train`` only and then applied to every split, so no
    statistic of the validation or test data leaks into training (BUG-04; previously they
    were computed on train + test).

    Parameters
    ----------
    y_train, y_val, y_test : ndarray, shape (n_samples, 4, 1)
        Descriptor targets ``[T30, C50, C80, D50]`` of each split (pre-normalization).

    Returns
    -------
    y_train, y_val, y_test : ndarray
        Normalized targets, same shapes as the inputs.
    T30_perc_95, C50_perc_95, C80_perc_95, D50_perc_95 : float
        Training 95th percentiles used for normalization (needed to de-normalize).
    """
    y_train = np.asarray(y_train, dtype=float)
    if y_train.ndim != 3 or y_train.shape[1:] != (4, 1):
        raise ValueError(f"`y_train` must have shape (n, 4, 1), got {y_train.shape}")

    norm = np.percentile(y_train[:, :, 0], 95, axis=0).reshape(1, 4, 1)

    def apply(y):
        return np.asarray(y, dtype=float) / norm

    T30_perc_95, C50_perc_95, C80_perc_95, D50_perc_95 = norm.ravel()
    return (apply(y_train), apply(y_val), apply(y_test),
            T30_perc_95, C50_perc_95, C80_perc_95, D50_perc_95)


def prediction(blind_estimation_model, X_test: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    """
    Predict the whole test set in batches, without rounding (TRAIN-07).

    The v0 version called ``predict`` once per sample (~25 min per band for ~17,500
    records) and rounded the outputs to 2 decimals in the normalized space.

    Parameters
    ----------
    blind_estimation_model : tensorflow.keras.Model
        Trained model instance (anything with a Keras-like ``predict``).
    X_test : ndarray, shape (n_samples, n_frames, 1)
        Test inputs (TAE).
    batch_size : int, default=1024
        Batch size passed to ``predict``.

    Returns
    -------
    ndarray, shape (n_samples, 4)
        Normalized predictions ``[T30, C50, C80, D50]``, unrounded.

    Raises
    ------
    ValueError
        If ``X_test`` is not 3-D or the model does not return one 4-vector per sample.
    """
    X_test = np.asarray(X_test)
    if X_test.ndim != 3:
        raise ValueError(f"`X_test` must have shape (n, n_frames, 1), got {X_test.shape}")
    pred = np.asarray(blind_estimation_model.predict(X_test, batch_size=batch_size, verbose=0))
    pred = pred.reshape(len(X_test), -1)
    if pred.shape[1] != 4:
        raise ValueError(f"Expected 4 outputs per sample, got shape {pred.shape}")
    return pred


def descriptors_err(prediction, y_test):
    """
    Element-wise prediction errors of each descriptor, in the normalized space.

    Parameters
    ----------
    prediction : array_like, shape (n_samples, 4)
        Model predictions (:func:`prediction`).
    y_test : ndarray, shape (n_samples, 4, 1) or (n_samples, 4)
        Normalized ground-truth targets.

    Returns
    -------
    err_t30, err_c50, err_c80, err_d50 : ndarray, shape (n_samples,)
        ``prediction - y_test`` per descriptor, unrounded (TRAIN-07).

    Raises
    ------
    ValueError
        If the shapes do not match.
    """
    pred = np.asarray(prediction, dtype=float).reshape(-1, 4)
    true = np.asarray(y_test, dtype=float).reshape(-1, 4)
    if pred.shape != true.shape:
        raise ValueError(f"`prediction` and `y_test` differ in shape: {pred.shape} vs {true.shape}")
    err = pred - true
    return err[:, 0], err[:, 1], err[:, 2], err[:, 3]


def save_exp_data(exp_num, band, blind_estimation_model, history, prediction,
                  err_t30, err_c50, err_c80, err_d50,
                  T30_perc_95, C50_perc_95, C80_perc_95, D50_perc_95,
                  X_test, y_test, test_metadata=None, training_seed=None):
    """
    Persist experiment artifacts (weights, training history, predictions, errors).

    Creates a results directory if needed (``results/exp{exp_num}``), saves the
    trained model weights for the specified band, and writes a pickle file with
    training curves, best epoch statistics, predictions, errors, normalization
    factors, and test data.

    Parameters
    ----------
    exp_num : int
        Experiment identifier (used in output path).
    band : int or str
        Frequency band label used in filenames.
    blind_estimation_model : tensorflow.keras.Model
        Trained model whose weights will be saved.
    history : tensorflow.keras.callbacks.History
        Training history object from ``model.fit``.
    prediction : sequence
        Model predictions for the test set.
    err_t30, err_c50, err_c80, err_d50 : sequence
        Per‑sample descriptor errors.
    T30_perc_95, C50_perc_95, C80_perc_95, D50_perc_95 : float
        Normalization scalars used during training.
    X_test, y_test : ndarray
        Held‑out test inputs/targets.
    test_metadata : pandas.DataFrame, optional
        One row per test record, in the order of ``X_test`` (TRAIN-08): at least the
        columns of :func:`code.metrics.record_metadata` (recording, speaker, rir, room,
        rir_type, augmentation), plus DRR/SNR columns when available. Stored under
        ``'test_metadata'`` so metrics can be broken down by RIR type, room or speaker.

    training_seed : int, optional
        TensorFlow seed of this run (TRAIN-09), stored under ``'training_seed'``.

    Returns
    -------
    None
    """
    # Create experiment results directory if it doesn't exist.
    isExist = os.path.exists(f'results/exp{exp_num}')
    if not isExist:
        os.makedirs(f'results/exp{exp_num}')

    # Save trained model weights for this band.
    blind_estimation_model.save_weights(f'results/exp{exp_num}/weights_{band}.weights.h5')

    # Determine best epoch (minimum validation loss).
    best_epoch = np.argmin(history.history['val_loss']) + 1

    # Collect results in a dictionary.
    results_dic = {'loss': history.history['loss'],
                   'val_loss': history.history['val_loss'],
                   'best_epoch': best_epoch,
                   'best_val_loss': min(history.history['val_loss']),
                   'total_epochs': len(history.history['loss']),
                   'prediction': prediction,
                   'err_t30': err_t30,
                   'err_c50': err_c50,
                   'err_c80': err_c80,
                   'err_d50': err_d50,
                   'T30_perc_95': T30_perc_95,
                   'C50_perc_95': C50_perc_95,
                   'C80_perc_95': C80_perc_95,
                   'D50_perc_95': D50_perc_95,
                   'X_test': X_test,
                   'y_test': y_test,
                   'training_seed': training_seed}
    if test_metadata is not None:
        if len(test_metadata) != len(y_test):
            raise ValueError(f"`test_metadata` has {len(test_metadata)} rows "
                             f"for {len(y_test)} test records")
        results_dic['test_metadata'] = test_metadata.reset_index(drop=True)

    with open(f'results/exp{exp_num}/results_{band}.pickle', 'wb') as handle:
        pickle.dump(results_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'Results saved in folder: results/exp{exp_num}')