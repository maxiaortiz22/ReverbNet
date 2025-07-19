"""
Experiment 4 configuration: train the model on the *entire* dataset **with pink
noise augmentation enabled** (``add_noise = True``).

RIR inclusion criterion: a room impulse response is considered valid if its
noise floor is at least 45 dB below the direct sound
(``max_ruido_dB = -45``).

Training signals will be mixed with pink noise over an SNR range of -5 dB to
+20 dB (see ``snr`` below).

Neural network hyperparameters used in this experiment:

    * filters = [32, 18, 8, 4]
    * kernel_size = [10, 5, 5, 5]
    * activation = ['relu','relu','relu','relu']
    * pool_size = [2,2,2]
    * learning_rate = 0.001

This module is imported as a config object; all variables below are read by the
training pipeline.
"""
import random
import os

# Global configuration:
seed = 2222  # Random seed initializer.
exp_num = 4  # Experiment identifier.

# Splits of RIRs for training and testing:
train = 0.8
test = 0.2

# Data:
tot_rirs_from_data = len(os.listdir('data/RIRs'))  # Total RIRs found under data/RIRs.
tot_to_augmentate = 15  # Number of RIRs to augment per room.
random.seed(seed)  # Seed to make random RIR selection reproducible.

# Parameters for descriptor calculation:
files_rirs = os.listdir('data/RIRs')  # RIR audio filenames.
files_rirs = random.sample(files_rirs, k=tot_rirs_from_data)

great_hall_rirs = [audio for audio in files_rirs if 'great_hall' in audio]
octagon_rirs = [audio for audio in files_rirs if 'octagon' in audio]
classroom_rirs = [audio for audio in files_rirs if 'classroom' in audio]

to_augmentate = []

# Select RIRs to augment from each room category.
for room in [great_hall_rirs, octagon_rirs, classroom_rirs]:
    to_augmentate.extend(random.sample(room, k=tot_to_augmentate))

sinteticas_rirs = [audio for audio in files_rirs if 'sintetica' in audio]
tot_sinteticas = len(sinteticas_rirs)

# Split synthetic RIRs:
aux_sinteticas_training = random.sample(sinteticas_rirs, k=int(tot_sinteticas*train))
aux_sinteticas_testing = [audio for audio in sinteticas_rirs if audio not in aux_sinteticas_training]

# Split augmented RIRs:
aux_aumentadas_training = random.sample(to_augmentate, k=int(len(to_augmentate)*train))
aux_aumentadas_testing = [audio for audio in to_augmentate if audio not in aux_aumentadas_training]

# Split real (non-synthetic, non-augmented) RIRs:
already_selected = sinteticas_rirs + to_augmentate
not_selected = [audio for audio in files_rirs if audio not in already_selected]

aux_reales_training = random.sample(not_selected, k=int(len(not_selected)*train))
aux_reales_testing = [audio for audio in not_selected if audio not in aux_reales_training]

# Final train/test RIR sets:
rirs_for_training = aux_sinteticas_training + aux_aumentadas_training + aux_reales_training
rirs_for_testing = aux_sinteticas_testing + aux_aumentadas_testing + aux_reales_testing

# Speech files:
files_speech_train = os.listdir('data/Speech/train')  # Training speech audio filenames.
files_speech_test = os.listdir('data/Speech/test')    # Test speech audio filenames.

# Analysis bands & filtering parameters:
bands = [125, 250, 500, 1000, 2000, 4000, 8000]  # Bands to analyze.
filter_type = 'octave band'  # Filter type: 'octave band' or 'third octave band'.
fs = 16000  # Audio sampling rate (Hz).
order = 4  # Filter order.
max_ruido_dB = -45  # Noise acceptance criterion (dB) to validate RIRs.
add_noise = True  # Enable pink noise augmentation for the dataset.
snr = [-5, 20]  # Target SNR range (dB) when adding noise.
tr_aug = [0.2, 3.1, 0.1]  # TR augmentation sweep: start, stop, step (s).
drr_aug = [-6, 19, 1]  # DRR augmentation sweep: start, stop, step (dB).

# Parameters for reading the generated database:
sample_frac = 1.0  # Fraction of the data to load.
random_state = seed  # Random seed passed downstream.

# Model hyperparameters:
filters = [32, 18, 8, 4]
kernel_size = [10, 5, 5, 5]
activation = ['relu','relu','relu','relu']
pool_size = [2,2,2]
learning_rate = 0.001

# Training parameters:
validation_split = 0.1 
batch_size = 1024
epochs = 500