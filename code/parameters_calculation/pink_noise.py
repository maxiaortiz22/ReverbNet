import colorednoise as cn
import numpy as np

def pink_noise(samples):
    #input values
    beta = 1 # the exponent: 0=white noite; 1=pink noise;  2=red noise (also "brownian noise")

    #Get noise:
    noise = cn.powerlaw_psd_gaussian(beta, samples)

    return noise / np.max(np.abs(noise))