import numpy as np
from librosa import resample
from scipy.signal import hilbert, sosfilt

#Función principal: TAE por frecuencia
def TAE(data, fs, sos_lowpass_filter):
    # Filtro la señal por banda de octava entre 125 y 4k Hz

    #Transformada de Hilbert
    analytic_signal = hilbert(data)
    #Obtengo la envolvente
    amplitude_envelope = np.abs(analytic_signal)

    #Obtengo la TAE a partir del resampleo y filtrado de la señal:
    tae = sosfilt(sos_lowpass_filter, amplitude_envelope) # Filtro pasabajos para la envolvente de la TAE

    #Downsampleo la señal:
    tae = resample(tae, orig_sr=fs, target_sr=40)

    tae = tae/np.max(np.abs(tae))

    #Si el audio tiene 5 [s] de duración y 16k de frec de sampleo, el tae debería ser de 200!
    assert(tae.size==200)

    return tae