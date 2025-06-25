import numpy as np
from progress.bar import IncrementalBar
import os
import pickle

def model(filters: list = [32, 18, 8, 4], 
           kernel_size: list = [10, 5, 5, 5], 
           activation: list = ['relu','relu','relu','relu'], 
           pool_size: list = [2,2,2],
           learning_rate: float = 0.001):

    """Se utiliza como modelo una red neruronal convolucional de 4 capas con los siguientes
    valores por defecto:
    1) filters = [32, 18, 8, 4], 
    2) kernel_size = [10, 5, 5, 5], 
    3) activation = ['relu','relu','relu','relu'], 
    4) pool_size = [2,2,2],
    5) learning_rate = 0.001
    """

    import tensorflow as tf
    import tensorflow.keras.layers as tfkl

    if len(filters) != 4 or len(kernel_size) != 4 or len(activation) != 4 or len(pool_size) != 3:
        raise ValueError("filters, kernel_size, and activation must have length 4; pool_size must have length 3")
    
    tf.keras.backend.clear_session()

    audio_in = tfkl.Input((200,1), name = 'Audio de entrada')

    capa_1 = tfkl.Conv1D(filters=filters[0], kernel_size=(kernel_size[0]), activation=activation[0], name='capa1_conv')(audio_in)
    capa_1 = tfkl.MaxPool1D(pool_size=pool_size[0], name='capa1_pool')(capa_1)
    capa_1 = tfkl.BatchNormalization(name = 'Batch_capa1')(capa_1)

    capa_2 = tfkl.Conv1D(filters=filters[1], kernel_size=(kernel_size[1]), activation=activation[1], name='capa2_conv')(capa_1) 
    capa_2 = tfkl.MaxPool1D(pool_size=pool_size[1], name='capa2_pool')(capa_2)
    capa_2 = tfkl.BatchNormalization(name = 'Batch_capa2')(capa_2)
    capa_2 = tfkl.Dropout(0.4, name='capa2_drop')(capa_2)

    capa_3 = tfkl.Conv1D(filters=filters[2], kernel_size=(kernel_size[2]), activation=activation[2], name='capa3_conv')(capa_2) 
    capa_3 = tfkl.MaxPool1D(pool_size=pool_size[2], name='capa3_pool')(capa_3)
    capa_3 = tfkl.BatchNormalization(name = 'Batch_capa3')(capa_3)
    
    capa_4 = tfkl.Conv1D(filters=filters[3], kernel_size=(kernel_size[3]), activation=activation[3], name='capa4_conv')(capa_3)
    capa_4 = tfkl.Flatten()(capa_4)

    tr_pred = tfkl.Dense(4, name='Salida_prediccion')(capa_4)
    
    modelo = tf.keras.Model(inputs=[audio_in], outputs=[tr_pred])
    modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                   loss='mse')
    return modelo

def reshape_data(tae, descriptors):
    """Función para cambiar las dimensiones de los datos para que puedan ser interpretados por Tensorflow.
    """
    tae_list = [[] for _ in range(len(tae))] #Genero una lista de listas vacías
    descriptors_list = [[] for _ in range(len(descriptors))] #Genero una lista de listas vacías

    for i in range(len(tae)):
        tae_list[i] = np.array(tae[i]).reshape(-1, 1)
        descriptors_list[i] = np.array(descriptors[i]).reshape(-1, 1)

    X = np.array(tae_list)
    y = np.array(descriptors_list)

    return X, y

def normalize_descriptors(descriptors, y_train, y_test):
    """Luego de hacer el reshape, se usa esta función para normalizar los descriptores
    con su percentil 95"""

    descriptors = list(descriptors)

    #Normalización de los parámetros:
    T30 = [descriptors[i][0][0] for i in range(len(descriptors))]
    C50 = [descriptors[i][1][0] for i in range(len(descriptors))]
    C80 = [descriptors[i][2][0] for i in range(len(descriptors))]
    D50 = [descriptors[i][3][0] for i in range(len(descriptors))]

    T30_perc_95 = np.percentile(T30, 95)
    C50_perc_95 = np.percentile(C50, 95)
    C80_perc_95 = np.percentile(C80, 95)
    D50_perc_95 = np.percentile(D50, 95)

    norm = np.array([T30_perc_95, C50_perc_95, C80_perc_95, D50_perc_95]).reshape(-1, 1)

    y_train = np.array([y_train[i]/norm for i in range(len(y_train))])
    y_test = np.array([y_test[i]/norm for i in range(len(y_test))])

    return y_train, y_test, T30_perc_95, C50_perc_95, C80_perc_95, D50_perc_95

def prediction(blind_estimation_model, X_test, y_test):
    """Función para calcular las predicciones del set de pruebas
    blind_estimation_model: Instancia de entrenamiento del modelo.
    X_test: set de pruebas de los TAE.
    y_test: set de pruebas de los descriptores."""
    
    prediction = []
    bar = IncrementalBar('Predicting values', max = int(len(y_test)))

    for i in range(len(y_test)):
        prediction.append(np.round(blind_estimation_model.predict(X_test[i,:,0].reshape(1,-1,1)),2)[0])
        bar.next()
    bar.finish()

    return prediction

def descriptors_err(prediction, y_test):
    """Función para calcular el error en la predicción de los descriptores."""

    err_t30, err_c50, err_c80, err_d50 = [], [], [], []
    bar = IncrementalBar('Calculating descriptors errors', max = int(len(y_test)))

    for i in range(len(y_test)):
        err_t30.append(np.round(prediction[i][0] - np.round(y_test[i,:,0].reshape(1,-1,1),2).flatten()[0],2))
        err_c50.append(np.round(prediction[i][1] - np.round(y_test[i,:,0].reshape(1,-1,1),2).flatten()[1],2))
        err_c80.append(np.round(prediction[i][2] - np.round(y_test[i,:,0].reshape(1,-1,1),2).flatten()[2],2))
        err_d50.append(np.round(prediction[i][3] - np.round(y_test[i,:,0].reshape(1,-1,1),2).flatten()[3],2))
        bar.next()
    bar.finish()

    return err_t30, err_c50, err_c80, err_d50

def save_exp_data(exp_num, band, blind_estimation_model, history, prediction, 
                  err_t30, err_c50, err_c80, err_d50, 
                  T30_perc_95, C50_perc_95, C80_perc_95, D50_perc_95,
                  X_test, y_test):
    """Función para guardar todos los datos del experimento para poder hacer el análisis en los notebooks"""

    #Creo el directorio donde se va a guardar el experimento si no existe:
    isExist = os.path.exists(f'results/exp{exp_num}')
    if not isExist:
        os.makedirs(f'results/exp{exp_num}')

    #Guardo los pesos del modelo entrenado:
    blind_estimation_model.save_weights(f'results/exp{exp_num}/weights_{band}.h5')

    #Guardo en un diccionario los resultados del modelo:
    results_dic = {'loss': history.history['loss'],
                   'val_loss': history.history['val_loss'],
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

    print(f'Resultados guardados en la carpeta: results/exp{exp_num}')