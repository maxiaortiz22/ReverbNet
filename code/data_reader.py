import pandas as pd
import os
from progress.bar import IncrementalBar

def read_dataset(band, db_name, sample_frac=1.0, random_state=None, type_data='train'):
    """Script para leer la base de datos:
    band (int): banda a leer.
    max_ruido_dB (int): máximo ruido permitido en la base de datos a leer.
    noise (bool): 'True' si se quiere leer la base de datos con ruido, sino 'False'.
    sample_frac (float): valor entre 0 y 1 que indica el porcentaje a leer de la base de datos.
    random_state (int): seed para poder reproducir un random de lectura de la base de datos."""

    partitions = os.listdir(f'cache/{db_name}')

    bar = IncrementalBar('Reading data base', max = len(partitions))

    db = pd.DataFrame()
    for partition in partitions:
        #Leo la base de datos:
        aux_df = pd.read_pickle(f'cache/{db_name}/{partition}')
        #Filtro por la banda y la fracción de datos que quiero:
        db = db.append(aux_df.loc[(aux_df.banda == band) & (aux_df.type_data == type_data)], ignore_index=True)
        bar.next()
    
    db = db.sample(frac=sample_frac, random_state=random_state)
    bar.finish()

    return db

