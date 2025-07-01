import sys; sys.path.append('code')
import os
import argparse
from code import import_configs_objs
from code import DataBase
# from code import read_dataset # <--- MODIFICACIÓN: Ya no usamos esta función
import pandas as pd # <--- MODIFICACIÓN: Importamos pandas para leer el HDF5
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
    parser.add_argument(
        "--save_batch_size", type=int, default=50000,
        help="Tamaño del lote para guardado incremental (default: 50000)")
    command_line_args = vars(parser.parse_args())
    return command_line_args

def main(**kwargs):
    """Función principal"""
    config_path = kwargs.pop("config")
    save_batch_size = kwargs.pop("save_batch_size", 50000)  # Tamaño del lote para guardado incremental
    
    print(f"Config: {config_path}")
    print(f"Tamaño de lote para guardado incremental: {save_batch_size}")
    
    config = import_configs_objs(config_path)

    database = DataBase(config['files_speech_train'], config['files_speech_test'], config['files_rirs'], config['tot_sinteticas'], config['to_augmentate'], 
                        config['rirs_for_training'], config['rirs_for_testing'], config['bands'], config['filter_type'], config['fs'], config['max_ruido_dB'], 
                        config['order'], config['add_noise'], config['snr'], config['tr_aug'], config['drr_aug'], batch_size=save_batch_size)

    db_name = database.get_database_name()

    # <--- MODIFICACIÓN: La comprobación de existencia ahora se hace dentro de la clase DataBase
    if database._database_exists():
        print('Base de datos calculada')
    else:
        print(f'Procesando {len(config["files_rirs"])} RIRs con guardado incremental...')
        with concurrent.futures.ProcessPoolExecutor() as executor:
            list(
                tqdm(
                    executor.map(database.calc_database_multiprocess, config['files_rirs']),
                    total=len(config['files_rirs']),
                    desc="Procesando RIRs"
                )
            )
        
        print("Combinando lotes temporales...")
        database._combine_temp_batches() # <--- MODIFICACIÓN: Esta llamada ahora ejecuta la combinación a HDF5
        print("Procesamiento completado!")
    
    del database
    gc.collect()

    # <--- INICIO DE BLOQUE MODIFICADO
    # --- Lectura de datos desde el archivo HDF5 ---
    print("\nCargando base de datos desde el archivo HDF5...")
    db_path = f'cache/{db_name}/database.h5'
    try:
        full_db = pd.read_hdf(db_path, key='data')
        print(f"Base de datos cargada con {len(full_db)} registros.")
    except Exception as e:
        print(f"No se pudo leer el archivo HDF5 en {db_path}. Error: {e}")
        return
    # --- FIN DE LECTURA DE DATOS ---

    # Entrenamiento:
    for band in config['bands']:
        print(f'\nInicio entrenamiento de la banda {band} Hz:')
        
        # --- Filtrado de datos para la banda y tipo de data actual ---
        # Reemplaza la funcionalidad de la antigua función 'read_dataset'
        print(f"Filtrando datos para la banda {band} Hz...")
        
        db_train_full = full_db[(full_db['banda'] == band) & (full_db['type_data'] == 'train')]
        db_test_full = full_db[(full_db['banda'] == band) & (full_db['type_data'] == 'test')]

        # Aplicar muestreo si es necesario
        if config['sample_frac'] < 1.0:
            db_train = db_train_full.sample(frac=config['sample_frac'], random_state=config['random_state'])
            db_test = db_test_full.sample(frac=config['sample_frac'], random_state=config['random_state'])
        else:
            db_train = db_train_full
            db_test = db_test_full
        
        print(f"Muestras de entrenamiento: {len(db_train)}, Muestras de prueba: {len(db_test)}")

        if db_train.empty or db_test.empty:
            print(f"No hay suficientes datos para la banda {band}. Saltando...")
            continue
        # <--- FIN DE BLOQUE MODIFICADO

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
        
        checkpoint_dir = f'results/exp{config["exp_num"]}/checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_filepath = os.path.join(checkpoint_dir, f'weights_{band}_best.weights.h5')

        model_checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filepath, save_weights_only=True, monitor='val_loss',
            mode='min', save_best_only=True, verbose=1)

        early_stopping_callback = EarlyStopping(
            monitor='val_loss', patience=100, mode='min',
            restore_best_weights=True, verbose=1)

        if os.path.exists(checkpoint_filepath):
            print(f"Cargando pesos del checkpoint: {checkpoint_filepath} para la banda {band}")
            blind_estimation_model.load_weights(checkpoint_filepath)

        history = blind_estimation_model.fit(x = X_train, y = y_train, 
                                             validation_split = config['validation_split'], 
                                             batch_size = config['batch_size'],  # Batch size para entrenamiento del modelo
                                             epochs = config['epochs'],
                                             callbacks=[model_checkpoint_callback, early_stopping_callback])

        if os.path.exists(checkpoint_filepath):
            print(f"Cargando pesos del mejor checkpoint final: {checkpoint_filepath} para la banda {band}")
            blind_estimation_model.load_weights(checkpoint_filepath)

        predict = prediction(blind_estimation_model, X_test, y_test)
        err_t30, err_c50, err_c80, err_d50 = descriptors_err(predict, y_test)

        save_exp_data(config['exp_num'], band, blind_estimation_model, history, predict, 
                      err_t30, err_c50, err_c80, err_d50, 
                      T30_perc_95, C50_perc_95, C80_perc_95, D50_perc_95,
                      X_test, y_test)

        del db_train, db_test, tae_train, tae_test, descriptors, X_train, X_test, y_train, y_test
        del descriptors_train, descriptors_test, db_train_full, db_test_full
        del T30_perc_95, C50_perc_95, C80_perc_95, D50_perc_95
        del blind_estimation_model, history, predict, err_t30, err_c50, err_c80, err_d50
        gc.collect()
    
    del full_db
    gc.collect()


if __name__ == "__main__":
    kwargs = parse_args()
    main(**kwargs)