import sys; sys.path.append('code')
import os
import argparse
from code import import_configs_objs
from code import DataBase
from code import read_dataset
from sklearn.model_selection import train_test_split
from code import model, reshape_data, normalize_descriptors, prediction, descriptors_err, save_exp_data
import concurrent.futures
import gc
from numpy import concatenate
from warnings import filterwarnings
filterwarnings("ignore")

# Importar las librerías necesarias para el checkpointing
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tqdm import tqdm
from tensorflow.keras.callbacks import EarlyStopping

def parse_args():
    """Función para parsear los argumentos de línea de comando"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", help="Config file with the experiment configurations")
    command_line_args = vars(parser.parse_args())
    return command_line_args

def main(**kwargs):
    """Función principal"""
    config_path = kwargs.pop("config")
    print(config_path)
    config = import_configs_objs(config_path)

    database = DataBase(config['files_speech_train'], config['files_speech_test'], config['files_rirs'], config['tot_sinteticas'], config['to_augmentate'], 
                        config['rirs_for_training'], config['rirs_for_testing'], config['bands'], config['filter_type'], config['fs'], config['max_ruido_dB'], 
                        config['order'], config['add_noise'], config['snr'], config['tr_aug'], config['drr_aug'])

    db_name = database.get_database_name()

    db_exists = False
    for folder in os.listdir('cache/'):
        if db_name in folder:
            db_exists = True

    if db_exists:
        print('Base de datos calculada')
    else:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(
                tqdm(
                    executor.map(database.calc_database_multiprocess, config['files_rirs']),
                    total=len(config['files_rirs']),
                    desc="Procesando RIRs"
                )
            )
        database.save_database_multiprocess(results)
    
    del database
    gc.collect()
        
    # Entrenamiento:
    for band in config['bands']:
        print(f'\nInicio entrenamiento de la banda {band} Hz:')

        db_train = read_dataset(band, db_name, config['sample_frac'], config['random_state'], type_data='train')
        db_test = read_dataset(band, db_name, config['sample_frac'], config['random_state'], type_data='test')

        tae_train = list(db_train.tae.to_numpy())
        tae_test = list(db_test.tae.to_numpy())
        descriptors_train = list(db_train.descriptors.to_numpy())
        descriptors_test = list(db_test.descriptors.to_numpy())

        X_train, y_train = reshape_data(tae_train, descriptors_train)
        X_test, y_test = reshape_data(tae_test, descriptors_test)

        descriptors = concatenate((y_train, y_test), axis=0)
        y_train, y_test, T30_perc_95, C50_perc_95, C80_perc_95, D50_perc_95 = normalize_descriptors(descriptors, y_train, y_test)

        blind_estimation_model = model(config['filters'], config['kernel_size'], config['activation'], 
                                       config['pool_size'], config['learning_rate'])
        
        # --- Configuración del Checkpointing ---
        checkpoint_dir = f'results/exp{config["exp_num"]}/checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True) # Crea el directorio si no existe

        # Ruta del archivo del checkpoint. Guardamos los pesos específicos de cada banda.
        checkpoint_filepath = os.path.join(checkpoint_dir, f'weights_{band}_best.weights.h5')

        # Configura el callback ModelCheckpoint
        model_checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True, # Solo guarda los pesos del modelo
            monitor='val_loss',     # Monitorea la pérdida de validación
            mode='min',             # Guarda el modelo cuando la pérdida de validación es mínima
            save_best_only=True,    # Solo guarda el mejor modelo
            verbose=1               # Muestra mensajes cuando se guarda un checkpoint
        )

        # Callback de EarlyStopping
        early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            patience=100,
            mode='min',
            restore_best_weights=True,
            verbose=1
        )

        # --- Cargar pesos si existe un checkpoint para esta banda ---
        if os.path.exists(checkpoint_filepath):
            print(f"Cargando pesos del checkpoint: {checkpoint_filepath} para la banda {band}")
            blind_estimation_model.load_weights(checkpoint_filepath)
        # ----------------------------------------

        history = blind_estimation_model.fit(x = X_train, y = y_train, 
                                             validation_split = config['validation_split'], 
                                             batch_size = config['batch_size'], 
                                             epochs = config['epochs'],
                                             callbacks=[model_checkpoint_callback, early_stopping_callback]) # Agrega los callbacks aquí

        # --- Cargar los mejores pesos antes de guardar resultados ---
        if os.path.exists(checkpoint_filepath):
            print(f"Cargando pesos del mejor checkpoint final: {checkpoint_filepath} para la banda {band}")
            blind_estimation_model.load_weights(checkpoint_filepath)
        # -----------------------------------------------------------

        predict = prediction(blind_estimation_model, X_test, y_test)

        err_t30, err_c50, err_c80, err_d50 = descriptors_err(predict, y_test)

        save_exp_data(config['exp_num'], band, blind_estimation_model, history, predict, 
                      err_t30, err_c50, err_c80, err_d50, 
                      T30_perc_95, C50_perc_95, C80_perc_95, D50_perc_95,
                      X_test, y_test)

        del db_train, db_test, tae_train, tae_test, descriptors, X_train, X_test, y_train, y_test
        del descriptors_train, descriptors_test
        del T30_perc_95, C50_perc_95, C80_perc_95, D50_perc_95
        del blind_estimation_model, history, predict, err_t30, err_c50, err_c80, err_d50
        gc.collect()


if __name__ == "__main__":
    kwargs = parse_args()
    main(**kwargs)