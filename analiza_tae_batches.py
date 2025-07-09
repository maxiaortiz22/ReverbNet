import glob
import pandas as pd
import os

# Cambia este path si tu base de datos tiene otro nombre
temp_dir = "cache/base_de_datos_-60_noise_False_traug_0.2_3.1_0.1_drraug_-6_19_1_snr_-5_20_temp"

batch_files = sorted(glob.glob(os.path.join(temp_dir, "batch_*.pkl")))

print(f"Se encontraron {len(batch_files)} lotes temporales.")

# Diccionario para acumular tipos y errores por columna
tipos_col = {}
errores_col = {}

def analiza_valor(val, col, batch_file, idx):
    errores = 0
    tipo = type(val)
    # Acumula tipos
    if col not in tipos_col:
        tipos_col[col] = {}
    tipos_col[col][tipo] = tipos_col[col].get(tipo, 0) + 1

    # Analiza casos especiales
    if isinstance(val, list):
        if len(val) == 0:
            print(f"[{col}] Lote {batch_file} Fila {idx}: lista vacía")
            errores += 1
        elif any(isinstance(elem, list) for elem in val):
            print(f"[{col}] Lote {batch_file} Fila {idx}: lista anidada {val}")
            errores += 1
        elif not all(isinstance(elem, (float, int, str, type(None))) for elem in val):
            print(f"[{col}] Lote {batch_file} Fila {idx}: lista con elementos no numéricos/str/None {val}")
            errores += 1
    elif hasattr(val, 'tolist'):
        # Es un array numpy u objeto similar
        pass
    elif val is None:
        print(f"[{col}] Lote {batch_file} Fila {idx}: NoneType")
        errores += 1
    elif isinstance(val, (float, int, str)):
        # Números o strings sueltos
        pass
    else:
        print(f"[{col}] Lote {batch_file} Fila {idx}: tipo inesperado {type(val)}")
        errores += 1
    return errores

for batch_file in batch_files:
    try:
        df = pd.read_pickle(batch_file)
        for col in df.columns:
            if col not in errores_col:
                errores_col[col] = 0
            for idx, val in enumerate(df[col]):
                errores_col[col] += analiza_valor(val, col, batch_file, idx)
        if 'tae' in df.columns:
            longitudes = set(len(x) for x in df['tae'])
            if len(longitudes) > 1:
                print(f"[tae] Lote {batch_file}: longitudes distintas {longitudes}")
        if 'descriptors' in df.columns:
            longitudes = set(len(x) for x in df['descriptors'])
            if len(longitudes) > 1:
                print(f"[descriptors] Lote {batch_file}: longitudes distintas {longitudes}")
    except Exception as e:
        print(f"Error leyendo {batch_file}: {e}")

print("\nResumen de tipos encontrados por columna:")
for col, tipos in tipos_col.items():
    print(f"\nColumna '{col}':")
    for tipo, count in tipos.items():
        print(f"  {tipo}: {count} ocurrencias")
    print(f"  Total de posibles errores detectados en '{col}': {errores_col[col]}")