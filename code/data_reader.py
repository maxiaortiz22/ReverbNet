import pandas as pd
import os

def read_dataset(band, db_name, sample_frac=1.0, random_state=None, type_data='train'):
    """
    Script para leer la base de datos en formato HDF5 de manera eficiente.

    Args:
        band (int): Banda de frecuencia a leer.
        db_name (str): Nombre de la carpeta de la base de datos (ej: 'base_de_datos_...').
        sample_frac (float): Porcentaje del dataset a devolver (entre 0 y 1).
        random_state (int): Seed para la reproducibilidad del muestreo.
        type_data (str): Tipo de datos a leer, 'train' o 'test'.
    
    Returns:
        pandas.DataFrame: Un DataFrame con los datos filtrados y muestreados.
    """
    # --- MODIFICACIÓN: La ruta ahora apunta directamente al archivo HDF5 ---
    db_path = f'cache/{db_name}/database.h5'

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"El archivo de base de datos no fue encontrado en: {db_path}")

    # --- MODIFICACIÓN: Leer todo el DataFrame y filtrar en memoria (formato fixed) ---
    query = f"banda == {band} and type_data == '{type_data}'"
    print(f"Leyendo datos desde '{db_path}' (formato fixed, sin consulta HDF5)")

    try:
        db = pd.read_hdf(db_path, key='data')
        db = db.query(query)
    except Exception as e:
        print(f"Ocurrió un error al leer el archivo HDF5: {e}")
        return pd.DataFrame() # Devuelve un DataFrame vacío en caso de error

    # El muestreo se aplica después de que los datos ya han sido filtrados.
    if sample_frac < 1.0:
        db = db.sample(frac=sample_frac, random_state=random_state)

    print(f"Lectura completada. Se cargaron {len(db)} registros.")
    
    return db