import numpy as np
import pandas as pd

def _to_float_array(x, expect_len=None):
    """Return np.array(float) or None if conversion fails."""
    if x is None:
        return None
    # Algunas filas pueden venir ya como np.ndarray; otras como list; otras como scalar.
    try:
        arr = np.asarray(x, dtype=float).ravel()
    except Exception:
        return None
    if expect_len is not None and arr.size != expect_len:
        # Mantengo el array (para poder inspeccionar), pero marco que la longitud no coincide
        # devolviendo igualmente el array; el control de longitud será externo.
        pass
    return arr

def check_db(df, descriptors_len=4, tae_min_len=1, verbose=True, sample=5):
    problems = {}

    # --- descriptors ---
    desc_arrs = df['descriptors'].map(lambda x: _to_float_array(x, expect_len=descriptors_len))
    desc_bad_type = desc_arrs.isna()  # conversion falló o valor None
    desc_bad_len  = desc_arrs.map(lambda a: (a is None) or (a.size != descriptors_len))
    desc_bad_finite = desc_arrs.map(lambda a: True if a is None else ~np.isfinite(a).all())
    problems['descriptors'] = desc_bad_type | desc_bad_len | desc_bad_finite

    # --- drr ---
    drr_num = pd.to_numeric(df['drr'], errors='coerce')
    drr_bad = drr_num.isna() | ~np.isfinite(drr_num.to_numpy())
    problems['drr'] = drr_bad

    # --- tae ---
    tae_arrs = df['tae'].map(lambda x: _to_float_array(x))
    tae_bad_type = tae_arrs.isna()
    tae_bad_empty = tae_arrs.map(lambda a: (a is None) or (a.size < tae_min_len))
    tae_bad_finite = tae_arrs.map(lambda a: True if a is None else ~np.isfinite(a).all())
    problems['tae'] = tae_bad_type | tae_bad_empty | tae_bad_finite

    # --- snr ---
    # Nota: NaN en snr es válido cuando no se agregó ruido. Lo registramos aparte.
    snr_num = pd.to_numeric(df['snr'], errors='coerce')
    snr_nan_valid = snr_num.isna()  # marcar pero no necesariamente “malo”
    snr_inf = ~np.isfinite(snr_num.fillna(0).to_numpy())  # ±Inf
    problems['snr'] = snr_inf  # sólo tratamos ±Inf como problema por defecto

    # Combinar
    bad_df = pd.DataFrame(problems)
    bad_row = bad_df.any(axis=1)

    summary = {
        'total_rows': len(df),
        'bad_rows_total': int(bad_row.sum()),
        'bad_descriptors': int(problems['descriptors'].sum()),
        'bad_drr': int(problems['drr'].sum()),
        'bad_tae': int(problems['tae'].sum()),
        'bad_snr_inf': int(problems['snr'].sum()),
        'snr_nan_valid_count': int(snr_nan_valid.sum()),
    }
    
    if verbose:
        print("=== Summary of problematic values ===")
        for k,v in summary.items():
            print(f"{k}: {v}")
        if sample and summary['bad_rows_total']>0:
            print(f"\n--- Sample of bad rows (n={min(sample, summary['bad_rows_total'])}) ---")
            print(df.loc[bad_row].head(sample).to_string())

    
    # Devuelvo estructuras útiles
    return {
        'summary': summary,
        'bad_mask': bad_row,
        'bad_detail': bad_df,
        'snr_nan_mask': snr_nan_valid,
        'desc_arrays': desc_arrs,
        'tae_arrays': tae_arrs,
        'drr_num': drr_num,
        'snr_num': snr_num,
    }


if __name__ == '__main__':
    import os
    cache_dir = "cache/data_base_-60_noise_True_traug_0.2_3.1_0.1_drraug_-6_19_1_snr_-5_20"  # carpeta que contiene los .pkl particionados
    dfs = []
    for f in os.listdir(cache_dir):
        if f.endswith(".pkl"):
            dfs.append(pd.read_pickle(f"{cache_dir}/{f}"))
    df = pd.concat(dfs, ignore_index=True)


    # Ejemplo de uso (suponiendo df ya cargado):
    results = check_db(df)
    # df_clean = df.loc[~results['bad_mask']].copy()
    
