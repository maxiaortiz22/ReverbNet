from .colored_noise import powerlaw_psd_gaussian
import numpy as np

def pink_noise(samples):
    #input values
    beta = 1 # the exponent: 0=white noite; 1=pink noise;  2=red noise (also "brownian noise")

    #Get noise:
    noise = powerlaw_psd_gaussian(beta, samples)

    return noise / np.max(np.abs(noise))