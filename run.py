import sys; sys.path.append('code')
import os
import argparse
import pandas as pd
from code import import_configs_objs
from code import DataBaseGenerator
from code import read_dataset
from sklearn.model_selection import train_test_split
from code import model, reshape_data, normalize_descriptors, prediction, descriptors_err, save_exp_data, create_early_stopping_callback
import concurrent.futures
import gc
from numpy import concatenate, argmin
from warnings import filterwarnings
filterwarnings("ignore")

def parse_args():
    """Función para parsear los argumentos de línea de comando"""

    # Inicializo el argparse
    parser = argparse.ArgumentParser()

    # Lista de argumentos por línea de comando
    parser.add_argument(
        "--config", help="Config file with the experiment configurations")

    # Convierto a diccionario
    command_line_args = vars(parser.parse_args())
    return command_line_args


def main(**kwargs):
    """Función principal adaptada para multiprocesamiento."""

    # --- Carga de configuración (sin cambios) ---
    config_path = kwargs.pop("config")
    print(f"Cargando configuración desde: {config_path}")
    config = import_configs_objs(config_path)

    # --- Instancia de la clase refactorizada ---
    # Asegúrate que los nombres de los parámetros coincidan (e.g., tr_aug -> tr_aug_params)
    database = DataBaseGenerator(
        config['files_speech_train'], config['files_speech_test'], config['files_rirs'],
        config['to_augmentate'], config['rirs_for_training'],
        config['rirs_for_testing'], config['bands'], config['filter_type'], config['fs'],
        config['max_ruido_dB'], config['order'], config['add_noise'], config['snr'],
        config['tr_aug'], config['drr_aug']
    )

    # ---------------------------------------------------------------------------
    # Generación de la base de datos (con multiprocesamiento y guardado por baches)
    # ---------------------------------------------------------------------------

    # Carpeta donde vivirán los pickles numerados
    db_dir = database.cache_path / database.db_name      # p.e. cache/BD-TR08_SNR20
    chunk_size = 50_000                                 # registros por pickle

    if db_dir.exists() and any(db_dir.iterdir()):
        print(f'La base de datos ya existe en: {db_dir}')
    else:
        print('Iniciando generación de base de datos con multiprocesamiento...')
        db_dir.mkdir(parents=True, exist_ok=True)       # crea la carpeta si no existe

        chunk_records: list[dict] = []
        part_idx: int = 0

        with concurrent.futures.ProcessPoolExecutor() as executor:
            # process_single_rir() debe devolver una lista de dicts por RIR
            results_iter = executor.map(
                database.process_single_rir,
                config['files_rirs']
            )

            for record_list in results_iter:
                # Acumulamos los registros de esta RIR
                chunk_records.extend(record_list)

                # Mientras haya más de chunk_size → escribir y vaciar buffer
                while len(chunk_records) >= chunk_size:
                    df_part = pd.DataFrame(chunk_records[:chunk_size])
                    df_part.to_pickle(db_dir / f'{part_idx}.pkl')
                    print(f'  ▸ Guardado {part_idx}.pkl con {len(df_part)} filas')
                    part_idx += 1
                    chunk_records = chunk_records[chunk_size:]  # descartar lo ya escrito

        # Terminado el pool → guardar lo que quede
        if chunk_records:
            df_part = pd.DataFrame(chunk_records)
            df_part.to_pickle(db_dir / f'{part_idx}.pkl')
            print(f'  ▸ Guardado {part_idx}.pkl con {len(df_part)} filas residuales')

        if part_idx == 0 and not chunk_records:
            print("Advertencia: no se generó ningún registro; la BD está vacía.")
        else:
            print(f'✅ Base de datos generada en {db_dir} con {part_idx+1} archivos.')

            
    # --- Limpieza de memoria ---
    db_name = database.db_name      # <-- guardo el nombre para usarlo luego
    del database 
    gc.collect()

    # --- Bucle de entrenamiento ---
    for band in config['bands']:
        print(f'\nInicio entrenamiento de la banda {band} Hz:')

        #Leo la fracción de datos especificados para la banda seleccionada:
        db_train = read_dataset(band, db_name, config['sample_frac'], config['random_state'], type_data='train')
        db_test = read_dataset(band, db_name, config['sample_frac'], config['random_state'], type_data='test')

        tae_train = list(db_train.tae.to_numpy())
        tae_test = list(db_test.tae.to_numpy())
        descriptors_train = list(db_train.descriptors.to_numpy())
        descriptors_test = list(db_test.descriptors.to_numpy())

        #Separo en train y test y les doy formato:
        X_train, y_train = reshape_data(tae_train, descriptors_train)
        X_test, y_test = reshape_data(tae_test, descriptors_test)

        #Normalizo según el percentil 95 de cada descriptor:
        descriptors = concatenate((y_train, y_test), axis=0)
        y_train, y_test, T30_perc_95, C50_perc_95, C80_perc_95, D50_perc_95 = normalize_descriptors(descriptors, y_train, y_test)

        #Instancio el modelo:
        blind_estimation_model = model(config['filters'], config['kernel_size'], config['activation'], 
                                       config['pool_size'], config['learning_rate'])
        #blind_estimation_model.summary()

        #Creo el callback de early stopping:
        early_stopping_callback = create_early_stopping_callback(patience=100, monitor='val_loss', restore_best_weights=True)
        
        #Entreno el modelo:
        history = blind_estimation_model.fit(x = X_train, y = y_train, 
                                             validation_split = config['validation_split'], 
                                             batch_size = config['batch_size'], 
                                             epochs = config['epochs'],
                                             callbacks=[early_stopping_callback])
        
        #Informo sobre el entrenamiento:
        best_epoch = argmin(history.history['val_loss']) + 1
        print(f'Mejor época: {best_epoch} con val_loss: {min(history.history["val_loss"]):.6f}')
        print(f'Entrenamiento terminado en época: {len(history.history["loss"])}')

        #Realizo las predicciones del modelo:
        predict = prediction(blind_estimation_model, X_test, y_test)

        #Calculo del error de los descriptores:
        err_t30, err_c50, err_c80, err_d50 = descriptors_err(predict, y_test)

        #Guardo los datos del experimento:
        save_exp_data(config['exp_num'], band, blind_estimation_model, history, predict, 
                      err_t30, err_c50, err_c80, err_d50, 
                      T30_perc_95, C50_perc_95, C80_perc_95, D50_perc_95,
                      X_test, y_test)

        # Elimino estas variables de memoria:
        del db_train, db_test, tae_train, tae_test, descriptors, X_train, X_test, y_train, y_test
        del descriptors_train, descriptors_test
        del T30_perc_95, C50_perc_95, C80_perc_95, D50_perc_95
        del blind_estimation_model, history, predict, err_t30, err_c50, err_c80, err_d50
        gc.collect()


if __name__ == "__main__":

    # 1) Lectura de los argumentos
    kwargs = parse_args()

    # 2) Función principal
    main(**kwargs)