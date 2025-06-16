"""
Script para generar las RIR sintéticas y meterlas en la base de datos!
"""

import numpy as np
from random import randrange

def syntheticRIR(Rt, fs):
  # Reconstruir una RIR con el método de schroeder a partir de conocer el tiempo de reverberación

  t = np.arange(0, Rt+0.5, 1/fs)
  y = np.e**((-6.9*t)/Rt) # Creo una exponencial decreciente

  seed = randrange(int(2**32))
  
  np.random.seed(seed)
  n = np.random.normal(0, 1, y.shape) #Genero ruido blanco

  y = y*n #Multiplico el ruido por la señal

  return y/np.max(np.abs(y)), seed

if __name__ == '__main__':
    import soundfile as sf

    #Valores de TR entre 0.2 a 3 [s] con pasos de 0.1:
    TRs = np.array([1, 2, 3])
    fs = 16000

    for i, TR in enumerate(TRs):
        TR = np.round(TR, 2)

        RIR, seed = syntheticRIR(TR, fs) #Genero una RIR sintética
        sf.write(f'sintetica_Seed{seed}_Tr{TR}.wav', RIR, fs) #Guardo el audio
        print(f'Se generó la RIR sintética con seed {seed} y TR {TR}')