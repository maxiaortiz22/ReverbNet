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
from progress.bar import IncrementalBar
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


def normalize_descriptors(descriptors, y_train, y_test):
    """
    Normalize descriptor targets by their global 95th percentile values.

    The 95th percentile is computed across the concatenated descriptor set
    (train + test) for each acoustic metric (T30, C50, C80, D50), and each
    sample in ``y_train`` and ``y_test`` is divided by the corresponding
    percentile. The percentile values are returned for later denormalization
    or reporting.

    Parameters
    ----------
    descriptors : array_like
        Combined descriptor collection (train + test) prior to normalization.
    y_train : array_like
        Training descriptor targets (pre‑normalization).
    y_test : array_like
        Test descriptor targets (pre‑normalization).

    Returns
    -------
    y_train : ndarray
        Normalized training targets.
    y_test : ndarray
        Normalized test targets.
    T30_perc_95, C50_perc_95, C80_perc_95, D50_perc_95 : float
        95th percentile values used for normalization (per descriptor).
    """
    descriptors = list(descriptors)

    # Extract each descriptor column.
    T30 = [descriptors[i][0][0] for i in range(len(descriptors))]
    C50 = [descriptors[i][1][0] for i in range(len(descriptors))]
    C80 = [descriptors[i][2][0] for i in range(len(descriptors))]
    D50 = [descriptors[i][3][0] for i in range(len(descriptors))]

    # Percentile (95th) per descriptor.
    T30_perc_95 = np.percentile(T30, 95)
    C50_perc_95 = np.percentile(C50, 95)
    C80_perc_95 = np.percentile(C80, 95)
    D50_perc_95 = np.percentile(D50, 95)

    norm = np.array([T30_perc_95, C50_perc_95, C80_perc_95, D50_perc_95]).reshape(-1, 1)

    y_train = np.array([y_train[i]/norm for i in range(len(y_train))])
    y_test = np.array([y_test[i]/norm for i in range(len(y_test))])

    return y_train, y_test, T30_perc_95, C50_perc_95, C80_perc_95, D50_perc_95


def prediction(blind_estimation_model, X_test, y_test):
    """
    Generate per‑sample predictions for the test set with a progress bar.

    Parameters
    ----------
    blind_estimation_model : tensorflow.keras.Model
        Trained model instance.
    X_test : ndarray
        Test inputs of shape (n_samples, 200, 1).
    y_test : ndarray
        Test targets (unused except for length to drive the progress bar).

    Returns
    -------
    list of ndarray
        Rounded predictions (2 decimal places) per test sample.
    """
    prediction = []
    bar = IncrementalBar('Predicting values', max=int(len(y_test)))

    for i in range(len(y_test)):
        prediction.append(np.round(blind_estimation_model.predict(X_test[i,:,0].reshape(1,-1,1)), 2)[0])
        bar.next()
    bar.finish()
    return prediction


def descriptors_err(prediction, y_test):
    """
    Compute element‑wise prediction errors for each acoustic descriptor.

    Parameters
    ----------
    prediction : sequence
        Model predictions (list/array of 4‑element vectors).
    y_test : ndarray
        Ground‑truth descriptor targets of shape (n_samples, 4, 1).

    Returns
    -------
    err_t30, err_c50, err_c80, err_d50 : list of float
        Prediction minus ground‑truth error (rounded to 2 decimals) per sample.
    """
    err_t30, err_c50, err_c80, err_d50 = [], [], [], []
    bar = IncrementalBar('Calculating descriptor errors', max=int(len(y_test)))

    for i in range(len(y_test)):
        err_t30.append(np.round(prediction[i][0] - np.round(y_test[i,:,0].reshape(1,-1,1), 2).flatten()[0], 2))
        err_c50.append(np.round(prediction[i][1] - np.round(y_test[i,:,0].reshape(1,-1,1), 2).flatten()[1], 2))
        err_c80.append(np.round(prediction[i][2] - np.round(y_test[i,:,0].reshape(1,-1,1), 2).flatten()[2], 2))
        err_d50.append(np.round(prediction[i][3] - np.round(y_test[i,:,0].reshape(1,-1,1), 2).flatten()[3], 2))
        bar.next()
    bar.finish()
    return err_t30, err_c50, err_c80, err_d50


def save_exp_data(exp_num, band, blind_estimation_model, history, prediction,
                  err_t30, err_c50, err_c80, err_d50,
                  T30_perc_95, C50_perc_95, C80_perc_95, D50_perc_95,
                  X_test, y_test):
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
                   'y_test': y_test}

    with open(f'results/exp{exp_num}/results_{band}.pickle', 'wb') as handle:
        pickle.dump(results_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'Results saved in folder: results/exp{exp_num}')