import pandas as pd

df = pd.read_pickle('cache/base_de_datos_-60_noise_False_traug_0.2_3.1_0.1_drraug_-6_19_1_snr_-5_20_temp/batch_1568_0000.pkl')
df.to_hdf('test_batch.h5', key='data', format='fixed')