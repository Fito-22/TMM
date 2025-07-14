"""
TMM main module for QOSS project under Carlos Anton supervision

Created on Fall 2024 in UAM

@author on: Adolfo Menendez
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.interpolate as interpid
from scipy.optimize import curve_fit
from layers import *

def snellCalculator(nlist, theta_in):
    """
    Calculate the angle list using Snell's law.

    Parameters:
    n_list (np.array): Array of refractive indices.
    theta_in (float): Incident angle in radians.

    Returns:
    np.array: Array of angles.
    """

    return np.arcsin( np.sin(theta_in) / nlist)

def interfaceR(pol, ni, nf, thi, thf):
    """
    Calculate the reflection coefficient at an interface between two media for given polarization.

    Parameters:
    pol (int): Polarization state (1 for parallel, 0 for perpendicular).
    ni (float): Refractive index of the initial medium.
    nf (float): Refractive index of the final medium.
    thi (float): Incident angle in radians.
    thf (float): Refraction angle in radians.

    Returns:
    float: The reflection coefficient.
    """
    if pol == 1:
        # Parallel polarization
        return (ni * np.cos(thi) - nf * np.cos(thf)) / (ni * np.cos(thi) + nf * np.cos(thf))
    else:
        # Perpendicular polarization
        return (nf * np.cos(thi) - ni * np.cos(thf)) / (nf * np.cos(thi) + ni * np.cos(thf))

def interfaceT(pol, ni, nf, thi, thf):
    """
    Calculate the transmission coefficient at an interface between two media for given polarization.

    Parameters:
    pol (int): Polarization state (1 for parallel, 0 for perpendicular).
    ni (float): Refractive index of the initial medium.
    nf (float): Refractive index of the final medium.
    thi (float): Incident angle in radians.
    thf (float): Refraction angle in radians.

    Returns:
    float: The transmission coefficient.
    """
    if pol == 1:
        # Parallel polarization
        return 2 * ni * np.cos(thi) / (ni * np.cos(thi) + nf * np.cos(thf))
    else:
        # Perpendicular polarization
        return 2 * ni * np.cos(thi) / (nf * np.cos(thi) + ni * np.cos(thf))

def TrM(delta,r,t):
    return 1/t * np.array([[np.exp(-1j * delta), np.exp(-1j * delta)*r],
                               [r * np.exp(1j * delta), np.exp(1j * delta)]])


def structure(lambda_cav,lambda_0,N_top,N_bot,Lcav):
    return np.array(
                            SiO2TiO2BM(lambda_cav, lambda_0, N_top) +
                            CavSpacer(lambda_cav, lambda_0, Lcav) +
                            TiO2SiO2BM(lambda_cav, lambda_0, N_bot)
                            )

# Coherent TMM main calculator

def CohTMM(pol, theta_in, lambda_0, lambda_cav, N_top, N_bot, Lcav):
    """
    Calculate coherent transfer matrix method properties.

    Parameters:
    pol (int): Polarization state (1 for parallel, 0 for perpendicular).
    theta_in (float): Incident angle in radians.
    lambda_0 (float): Incident Wavelength.
    lambda_cav (float): Cavity Wavelength.
    N_top (int): Number of pairs at the top mirror.
    N_bot (int): Number of pairs at the bottom mirror.
    Lcav (float): Length between the mirrors in units of lambda_cav.

    Returns:
    List: Calculated properties [theta_in, lambda_0, Abs(Mtilde[2, 1]/Mtilde[1, 1])^2, Abs(1.0/Mtilde[1, 1])^2, 1239.87 * 0.001 / lambda_0]
    """

    # Definition of the structure. Use your own structure here.
    list_structure = structure(lambda_cav,lambda_0,N_top,N_bot,Lcav)

    n_list = np.concatenate([[1.0], list_structure[:, 1], [1.0]])

    angle_list = snellCalculator(n_list, theta_in)
    kz_list = 2.0 * np.pi * n_list * np.cos(angle_list) / lambda_0
    delta = kz_list[1:-1] * list_structure[:, 0]

    t_list = np.array([interfaceT(pol, n_list[i], n_list[i + 1], angle_list[i], angle_list[i + 1]) for i in range(len(n_list) - 1)])
    r_list = np.array([interfaceR(pol, n_list[i], n_list[i + 1], angle_list[i], angle_list[i + 1]) for i in range(len(n_list) - 1)])

    M_list = [TrM(delta[i-1], r_list[i], t_list[i]) for i in range(1, len(n_list) - 1)]
    M_product = np.linalg.multi_dot(M_list)

    M_tilde = 1.0/t_list[0] * np.dot(np.array([[1.0, r_list[0]], [r_list[0], 1.0]]), M_product)
    # r_final -> M_tilde[1, 0] / M_tilde[0, 0]
    # t_final -> 1.0 / M_tilde[0, 0]

    return [theta_in, lambda_0, np.abs(M_tilde[1, 0] / M_tilde[0, 0])**2, np.abs(1.0 / M_tilde[0, 0])**2, 1239.87 * 0.001 / lambda_0] # (1239.87 * 0.001 lambda to energy conversion)


def simulate_ref(pol, theta, lambda_cav, delta_lambda, Npoints,N_top, N_bot, Lcav, plot=False):
    """
    Calculate the reflection of the cavity at different lambda.

    Parameters:
    pol (int): Polarization state (1 for parallel, 0 for perpendicular).
    structure (np.array): Structure of materials as a 2D array.
    theta (float): Incident angle in radians.
    lambda_cav (float): Cavity Wavelength.
    delta_lambda (float): Wavelength range.
    Npoints (int): Number of points.
    N_top (int): Number of pairs at the top mirror.
    N_bot (int): Number of pairs at the bottom mirror.
    Lcav (float): Cavity length.
    plot (bool): Plot the results.

    Returns:
    Dictionary: Calculated properties including wavelengths, energies, and reflectivity.
    """

    wavelengths = np.linspace(lambda_cav-delta_lambda, lambda_cav+delta_lambda, Npoints)
    TMMcalculator = np.array([CohTMM(pol, theta, lambda_, lambda_cav, N_top, N_bot, Lcav) for lambda_ in wavelengths])

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(TMMcalculator[:, 1], TMMcalculator[:, 2], 'r-')
        plt.xlabel('Wavelength (μm)')
        plt.ylabel('Reflectivity')
        plt.title('Spectrum')
        plt.savefig("Reflectivity.png")
        plt.show()


    return {
        'plot': plt.gcf(),
        'wavelengths': TMMcalculator[:, 1],
        'R': TMMcalculator[:, 2],
        'energy': TMMcalculator[:, 4]
    }

def simulateQ(pol, theta, lambda_cav, delta_lambda, Npoints, N_top, N_bot, Lcav, plot=True):
    """
    Calculate the the quality factor of a cavity. Be aware that the peaks function may be considering peaks outside the sideband.

    Parameters:
    pol (int): Polarization state (1 for parallel, 0 for perpendicular).
    theta (float): Incident angle in radians.
    lambda_cav (float): Wavelength.
    delta_lambda (float): Wavelength range.
    Npoints (int): Number of points.
    N_top (int): Number of pairs at the top mirror.
    N_bot (int): Number of pairs at the bottom mirror.
    Lcav (float): Cavity length.
    plot (bool): Plot the results.

    Returns:
    Dictionary: Calculated properties including structure, linewidth, Q factor, and absorption.
    """

    # Simulate the spectrum
    wavelengths = np.linspace(lambda_cav - delta_lambda, lambda_cav + delta_lambda, Npoints)
    TMMcalculator = np.array([CohTMM(pol, theta, lambda_, lambda_cav, N_top, N_bot, Lcav) for lambda_ in wavelengths])

    # Calculate the Q factor
    y = TMMcalculator[:,2]
    valleys, _ = find_peaks(-y)
    half = peak_widths(x=-y,peaks=valleys, rel_height=0.5)[0]
    FWHM = half * (wavelengths[-1]-wavelengths[0])/Npoints
    Q = wavelengths[valleys]/FWHM

    if plot == True:
        # Plot the fit results
        plt.figure(figsize=(10, 6))
        plt.plot(TMMcalculator[:,4], TMMcalculator[:, 2], "b", label='Simulation Data')
        plt.scatter(TMMcalculator[valleys,4], TMMcalculator[valleys,2], color='red', label='peaks')
        plt.xlabel('Energy (eV)', fontsize = 30)
        plt.ylabel(r'R', fontsize = 30)
        plt.tick_params(axis='both', which='major', labelsize=30)
        # plt.legend(fontsize=20)
        plt.show()

    # Output results
    return {
        "plot": plt.gcf(),
        "Qfact":Q,
        "wavelengths": TMMcalculator[:, 1],
        "R": TMMcalculator[:, 2],
        "energy":TMMcalculator[:,4],
        "lambda_0": wavelengths[valleys],
        "energy_0":TMMcalculator[valleys,4],
        "R_peak": TMMcalculator[:, 2][valleys]
    }

def modesplitting(pol, theta, lambda_cav, delta_lambda, Npoints, N_top, N_bot, Lcav_list):
    """
    Calculate R for different distances of the cavity in units of lambda.

    Parameters:
    pol (int): Polarization state (1 for parallel, 0 for perpendicular).
    theta (float): Incident angle in radians.
    lambda_cav (float): Wavelength.
    delta_lambda (float): Wavelength range.
    Npoints (int): Number of points.
    N_top (int): Number of pairs at the top mirror.
    N_bot (int): Number of pairs at the bottom mirror.
    Lcav_list (list or array): List/array of cavity lengths in units of lambda.

    Returns:
    DataFrame with 3 columns: Energy, Lcav and R
    """

    # Initialize the lists for the loop
    energies = []
    lcavs = []
    Rs = []

    # Loop to simulate and save the results per Lcav
    for Lcav in Lcav_list:
        vector = simulate_ref(pol, theta, lambda_cav, delta_lambda, Npoints, N_top, N_bot, Lcav)
        energies.extend(vector["energy"])
        lcavs.extend([Lcav] * len(vector["energy"]))
        Rs.extend(vector["R"])

    # Create a DataFrame with the data
    df = pd.DataFrame({
        'Energy': energies,
        'Lcav': lcavs,
        'R': Rs
    })

    return df

def disprel(pol, theta_list, lambda_cav, delta_lambda, Npoints, N_top, N_bot, Lcav):
    """
    Calculate R for different angles (k-space).

    Parameters:
    pol (int): Polarization state (1 for parallel, 0 for perpendicular).
    theta_list (list or array): List/array of incident angle in radians.
    lambda_cav (float): Wavelength.
    delta_lambda (float): Wavelength range.
    Npoints (int): Number of points.
    N_top = N Pairs on the top mirror
    N_bot = N Pairs on the bottom mirror
    Lcav (float): Cavity length in units of lambda.

    Returns:
    DataFrame with 3 columns: Energy, k// and R
    """
    # Initialize the lists for the loop
    energies = []
    ks = []
    Rs = []

    # Loop to simulate and save the results per theta
    for theta in theta_list:
        vector = simulate_ref(pol, theta, lambda_cav, delta_lambda, Npoints, N_top, N_bot, Lcav)
        energies.extend(vector["energy"])
        ks.extend([2 * np.pi * np.sin(theta)/lambda_cav] * len(vector["energy"]))
        Rs.extend(vector["R"])

    # Create a DataFrame with the data
    df = pd.DataFrame({
        'Energy': energies,
        'k//': ks,
        'R': Rs
    })

    return df
