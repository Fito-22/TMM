"""
TMM module to define the layers of the structure

Created on Fall 2024 in UAM

@author on: Adolfo Menendez
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.interpolate as interpid
from scipy.optimize import curve_fit

# Import data reflexiv index for lambda from 0.3 to 1 um
SiO2_df = pd.read_csv('data_nk/SiO2 - thin film 2016.txt', sep='\t')
TiO2_df = pd.read_csv('data_nk/TiO2.txt', sep='\t')
MoSe2_df = pd.read_csv('data_nk/Hsu 2019- monolayer (1L) film.txt', sep='\t')
WS2_df = pd.read_csv('data_nk/WS2_Chernikov.txt', sep='\t')
Ag_df = pd.read_csv('data_nk/Ag_Ferrera.txt', sep='\s+')
PMMA_df = pd.read_csv('data_nk/PMMA.txt', sep='\s+')

# Correction for n_air
def n_air(lambda_):
    return 1 + 0.05792105/(238.0185 - lambda_**-2) + 0.00167917/(57.362 - lambda_**-2)

# refractive index from HBN and PMMA
def n_HBN(lambda_):
    return np.sqrt(1+3.263 * (lambda_ * 1000)**2/((lambda_ * 1000)**2 - 164.4**2))

# def n_PMMA(lambda_):
#     return np.sqrt(2.1778 + 6.1209*10**(-3) * lambda_**2 -
#                    1.5004 * 10**(-3) * lambda_**(4) +
#                    2.3678 *10**(-2)*lambda_**(-2) -
#                    4.2137*10**(-3)*lambda_**(-4) +
#                    7.3417 * 10**(-4)*lambda_**(-6) -
#                    4.5042*10**(-5)*lambda_**(-8))

# Interpolation functions
def function_n_SiO2(lambda_):
    f = interpid.interp1d(SiO2_df['lambda'], SiO2_df['n'], kind='linear')
    return f(lambda_)
def function_k_SiO2(lambda_):
    f = interpid.interp1d(SiO2_df['lambda'], SiO2_df['k'], kind='linear')
    return f(lambda_)
def function_n_TiO2(lambda_):
    f = interpid.interp1d(TiO2_df['lambda'], TiO2_df['n'], kind='linear')
    return f(lambda_)
def function_k_TiO2(lambda_):
    f = interpid.interp1d(TiO2_df['lambda'], TiO2_df['k'], kind='linear')
    return f(lambda_)
def function_n_MoSe2(lambda_):
    f = interpid.interp1d(MoSe2_df['lambda'], MoSe2_df['n'], kind='linear')
    return f(lambda_)
def function_k_MoSe2(lambda_):
    f = interpid.interp1d(MoSe2_df['lambda'], MoSe2_df['k'], kind='linear')
    return f(lambda_)
def function_n_WS2(lambda_):
    f = interpid.interp1d(WS2_df['lambda'], WS2_df['n'], kind='linear')
    return f(lambda_)
def function_k_WS2(lambda_):
    f = interpid.interp1d(WS2_df['lambda'], WS2_df['k'], kind='linear')
    return f(lambda_)
def function_n_Ag(lambda_):
    f = interpid.interp1d(Ag_df['lambda'], Ag_df['n'], kind='linear')
    return f(lambda_)
def function_k_Ag(lambda_):
    f = interpid.interp1d(Ag_df['lambda'], Ag_df['k'], kind='linear')
    return f(lambda_)
def function_n_PMMA(lambda_):
    f = interpid.interp1d(PMMA_df["lambda"], PMMA_df["n"], kind="linear")
    return f(lambda_)
def function_k_PMMA(lambda_):
    f = interpid.interp1d(PMMA_df["lambda"], PMMA_df["k"], kind="linear")
    return f(lambda_)



#############################################################################################
# Define the layers
def SiO2TiO2BM(lambda_cav, lambda_, Npairs):
    """
    Layer of SiO2 with TiO2 between them

    lambda_cav: central wavelength
    lambda_: incident wavelength
    Npairs: number of pairs of layers
    """
    delta_SiO2 = lambda_cav / (4 * function_n_SiO2(lambda_cav))
    delta_TiO2 = lambda_cav / (4 * function_n_TiO2(lambda_cav))
    n_SiO2 = function_n_SiO2(lambda_)
    n_TiO2 = function_n_TiO2(lambda_)
    k_SiO2 = function_k_SiO2(lambda_)
    k_TiO2 = function_k_TiO2(lambda_)

    # Generate the layer structure
    layers = []
    for i in range(Npairs):
        layers.append([delta_SiO2, n_SiO2 + 1j * k_SiO2])
        layers.append([delta_TiO2, n_TiO2 + 1j * k_TiO2])
    return layers

def profilerSiO2TiO2BM(lambda_cav, lambda_, Npairs,samples):
    """
    lambda_cav: central wavelength
    lambda_: incident wavelength
    Npairs: number of pairs of layers
    samples_: number of samples of the spectrum
    """

    # Layer of SiO2 with TiO2 between them
    delta_SiO2 = lambda_cav / (4 * samples * function_n_SiO2(lambda_cav))
    delta_TiO2 = lambda_cav / (4 * samples * function_n_TiO2(lambda_cav))
    n_SiO2 = function_n_SiO2(lambda_)
    n_TiO2 = function_n_TiO2(lambda_)
    k_SiO2 = function_k_SiO2(lambda_)
    k_TiO2 = function_k_TiO2(lambda_)

    # Generate the layer structure
    layer_1 = [[delta_SiO2, n_SiO2 + 1j* k_SiO2] for i in range(samples)]
    layer_2 = [[delta_TiO2, n_TiO2 + 1j* k_TiO2] for i in range(samples)]
    structure = []
    for i in range(Npairs):
        structure.extend(layer_1)
        structure.extend(layer_2)
    return structure

def TiO2SiO2BM(lambda_cav, lambda_, Npairs):
    """
    Layer of TiO2 with SiO2 between them

    lambda_cav: central wavelength
    lambda_: incident wavelength
    Npairs: number of pairs of layers
    """
    delta_SiO2 = lambda_cav / (4 * function_n_SiO2(lambda_cav))
    delta_TiO2 = lambda_cav / (4 * function_n_TiO2(lambda_cav))
    n_SiO2 = function_n_SiO2(lambda_)
    n_TiO2 = function_n_TiO2(lambda_)
    k_SiO2 = function_k_SiO2(lambda_)
    k_TiO2 = function_k_TiO2(lambda_)

    # Generate the layer structure
    layers = []
    for i in range(Npairs):
        layers.append([delta_TiO2, n_TiO2 + 1j * k_TiO2])
        layers.append([delta_SiO2, n_SiO2 + 1j * k_SiO2])
    return layers

def profilerTiO2SiO2BM(lambda_cav, lambda_, Npairs,samples):
    """
    lambda_cav: central wavelength
    lambda_: incident wavelength
    Npairs: number of pairs of layers
    samples_: number of samples of the spectrum
    """
    # Layer of TiO2 with SiO2 between them
    delta_SiO2 = lambda_cav / (4 * samples * function_n_SiO2(lambda_cav))
    delta_TiO2 = lambda_cav / (4 * samples * function_n_TiO2(lambda_cav))
    n_SiO2 = function_n_SiO2(lambda_)
    n_TiO2 = function_n_TiO2(lambda_)
    k_SiO2 = function_k_SiO2(lambda_)
    k_TiO2 = function_k_TiO2(lambda_)

    # Generate the layer structure
    layer_1 = [[delta_SiO2, n_SiO2 + 1j* k_SiO2] for i in range(samples)]
    layer_2 = [[delta_TiO2, n_TiO2 + 1j* k_TiO2] for i in range(samples)]
    structure = []
    for i in range(Npairs):
        structure.extend(layer_2)
        structure.extend(layer_1)
    return structure

def CavSpacer(lambda_cav, lambda_, Ncav):
    """
    lambda_cav: central wavelength
    lambda_: incident wavelength
    Ncav: lenght in units of lambda
    """
    # Layer of air
    return [[Ncav*lambda_cav/n_air(lambda_cav) ,n_air(lambda_)]]

def profilerCavSpacer(lambda_cav, lambda_, Ncav, samples):
    """
    lambda_cav: central wavelength
    lambda_: incident wavelength
    Ncav: lenght in units of lambda
    samples_: number of samples of the spectrum
    """
    # Layer of air
    return [[Ncav*lambda_cav/(samples*n_air(lambda_cav)) ,n_air(lambda_)] for i in range(samples)]

def SiO2layer(lambda_, delta_):
    """
    Layer of SiO2
    """
    return [[delta_, function_n_SiO2(lambda_) + 1j* function_k_SiO2(lambda_)]]

def profilerSiO2layer(lambda_, delta_, samples):
    # Layer of SiO2
    return [[delta_/samples, function_n_SiO2(lambda_) + 1j*function_k_SiO2(lambda_)] for i in range(samples)]

def TiO2layer(lambda_, delta_):
    # Layer of TiO2
    return [[delta_, function_n_TiO2(lambda_) + 1j*function_k_TiO2(lambda_)]]

def profilerTiO2layer(lambda_, delta_, samples):
    # Layer of TiO2
    return [[delta_/samples, function_n_TiO2(lambda_)+ 1j*function_k_TiO2(lambda_)] for i in range(samples)]

def ArbitraryLayer(delta_, n_):
    # Layer of arbitrary material
    return [[delta_, n_]]

def profilerArbitraryLayer(delta_, n_, samples):
    # Layer of arbitrary material
    return [[delta_/samples, n_] for i in range(samples)]

def PMMALayer(lambda_, delta_):
    return [[delta_, function_n_PMMA(lambda_)+1j*function_k_PMMA(lambda_)]]

def profilerPMMALayer(lambda_, delta_, samples):
    return [[delta_/samples, function_n_PMMA(lambda_)+ 1j*function_k_PMMA(lambda_)] for i in range(samples)]

def AgLayer(lambda_, delta_):
    return [[delta_, function_n_Ag(lambda_) + 1j*function_k_Ag(lambda_)]]

def profilerAgLayer(lambda_, delta_, samples):
    return [[delta_/samples, function_n_Ag(lambda_)+ 1j*function_k_Ag(lambda_)] for i in range(samples)]
