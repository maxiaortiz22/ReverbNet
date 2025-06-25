"""
Experimento 3: Entreno el modelo con toda la base de datos, agregando ruido rosa.

El criterio de aceptación de ruido para determinar si una RIR es válida o no será de 60 dB.

Los valores de SNR de las señales para entrenar van a estar entre: -5 y 20 dB

Para este caso uso los siguientes parámetros para la red:

    * filters = [32, 18, 8, 4] 
    * kernel_size = [10, 5, 5, 5] 
    * activation = ['relu','relu','relu','relu'] 
    * pool_size = [2,2,2]
    * learning_rate = 0.001
"""
import random
import os

#Configuración global:
seed = 2222 #Inicializador del generador de números random
exp_num = 3 #Número del experimento

#Splits of RIRs for training and testing:
train = 0.8
test = 0.2


# Data:
tot_rirs_from_data = len(os.listdir('data/RIRs')) #Cantidad de RIRs a agarrar de la carpeta data/RIRs
tot_to_augmentate = 15 #Elijo 15 audios de cada sala para aumentar.
random.seed(seed) #Seed para hacer reproducible el random de agarrado de RIRs

#Parámetros para el cálculo de los descriptores:
files_rirs = os.listdir('data/RIRs') #Audios de las RIRs
files_rirs = random.sample(files_rirs, k=tot_rirs_from_data)

great_hall_rirs = [audio for audio in files_rirs if 'great_hall' in audio]
octagon_rirs = [audio for audio in files_rirs if 'octagon' in audio]
classroom_rirs = [audio for audio in files_rirs if 'classroom' in audio]

to_augmentate = []

for room in [great_hall_rirs, octagon_rirs, classroom_rirs]:
    to_augmentate.extend(random.sample(room, k=tot_to_augmentate))

sinteticas_rirs = [audio for audio in files_rirs if 'sintetica' in audio]
tot_sinteticas = len(sinteticas_rirs)


#Separo las sintéticas:
aux_sinteticas_training = random.sample(sinteticas_rirs, k=int(tot_sinteticas*train))
aux_sinteticas_testing = [audio for audio in sinteticas_rirs if audio not in aux_sinteticas_training]

#Separo las aumentadas:
aux_aumentadas_training = random.sample(to_augmentate, k=int(len(to_augmentate)*train))
aux_aumentadas_testing = [audio for audio in to_augmentate if audio not in aux_aumentadas_training]

#Separo las reales:
already_selected = sinteticas_rirs + to_augmentate
not_selected = [audio for audio in files_rirs if audio not in already_selected]

aux_reales_training = random.sample(not_selected, k=int(len(not_selected)*train))
aux_reales_testing = [audio for audio in not_selected if audio not in aux_reales_training]


rirs_for_training = aux_sinteticas_training + aux_aumentadas_training + aux_reales_training
rirs_for_testing = aux_sinteticas_testing + aux_aumentadas_testing + aux_reales_testing


files_speech_train = os.listdir('data/Speech/train') #Audios de voz entrenamiento
files_speech_test = os.listdir('data/Speech/test') #Audios de voz prueba
bands = [125, 250, 500, 1000, 2000, 4000, 8000] #Bandas a analizar
filter_type = 'octave band' #Tipo de filtro a utilizar: 'octave band' o 'third octave band'
fs = 16000 #Frecuencia de sampleo de los audios.
order = 4 #Orden del filtro
max_ruido_dB = -60 #Criterio de aceptación de ruido para determinar si una RIR es válida o no
add_noise = True #Booleano para definir si agregar ruido rosa o no a la base de datos.
snr = [-5, 20] #Valores de SNR que tendrían los audios si se les agrega ruido
tr_aug = [0.2, 3.1, 0.1] #Aumentar los valores de TR de 0.2 a 3 s con pasos de 0.1 s
drr_aug = [-6, 19, 1] #Aumentar los valores de DRR de -6 a 18 dB con pasos de 1 dB

#Parámetros para la lectura de la base de datos:
sample_frac = 1.0 #Fracción de la data a leer
random_state = seed #Inicializador del generador de números random

#train = 0.8
#test = 0.2

# Modelo:

#Parámetros de la red:
filters = [32, 18, 8, 4] 
kernel_size = [10, 5, 5, 5] 
activation = ['relu','relu','relu','relu'] 
pool_size = [2,2,2]
learning_rate = 0.001

#Parámetros de entrenamiento:
validation_split = 0.1 
batch_size = 1024
epochs = 500