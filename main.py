"""
TMM main module for QOSS project under Carlos Anton supervision. This code calculates the Q factor for a different set of configurations.

Created on Fall 2024 in UAM

@author on: Adolfo Menendez
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.interpolate as interpid
from scipy.optimize import curve_fit
import time
from layers import *
from functions import *
import os


#Set the initial parameters lists
lambda_cav_list = [0.52, 0.65] # um
N_top_list = list(range(5,9)) + list(range(10,13))
N_bot_list = list(range(10,13))
tot = len(lambda_cav_list)*len(N_top_list)*len(N_bot_list)
print(tot)
###############################################################################################################################

# MAIN LOOP

#########################################################################################################################
it = 0

df = pd.DataFrame(columns=["lambda_cav", "Lcav", "N_top", "N_bot", "Q_fact", "peak"]).astype({"lambda_cav":"float64", "Lcav":"float64", "N_top":"int32", "N_bot":"int32", "Q_fact": "float64", "peak":"float64"})

for lambda_cav in lambda_cav_list:
    # Lcav_list = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10])*lambda_cav # um
    Lcav_list = np.array([1])*lambda_cav # um
    for N_top in N_top_list:
        for N_bot in N_bot_list:
            for Lcav in Lcav_list:
                ############################################################################################
                start_time = time.time()
                #Set the structure
                Ncav = Lcav/lambda_cav # um
                structure = np.array(
                                        SiO2layer(lambda_cav, 10) +
                                        SiO2TiO2BM(lambda_cav, lambda_cav, N_top) +
                                        SiO2layer(lambda_cav, lambda_cav/(4*function_n_SiO2(lambda_cav))) +
                                        CavSpacer(lambda_cav, lambda_cav, Ncav) +
                                        SiO2layer(lambda_cav, lambda_cav/(4*function_n_SiO2(lambda_cav))) +
                                        TiO2SiO2BM(lambda_cav, lambda_cav, N_bot) +
                                        SiO2layer(lambda_cav, 10)
                                    )

                # Create a folder
                # folder_name = f'Simulations/data_{lambda_cav}_{N_top}_{N_bot}_{Lcav}'
                # os.makedirs(folder_name, exist_ok=True)

                #################################################################################################
                # Creating the DBR

                # Rewrite and separate the structure profile to plot it
                depth_tab = np.insert(np.cumsum(structure[:, 0]), 0, 0)
                n_struct = []
                for i in range(len(structure)):
                    n_real = structure[i, 1].real
                    n_struct.append([depth_tab[i], n_real])

                # Extract depth and refractive index values
                depth = [point[0] for point in n_struct]
                refractive_index = [point[1] for point in n_struct]

                middle = depth[-1]/2
                # Create the plot
                plt.figure()

                # Plotting the data
                plt.step(depth, refractive_index, where='post', color=(0, 0.46, 0.86), label='Refractive index')
                # Filling under the plot
                plt.fill_between(depth, refractive_index, step='post', color=(0, 0.46, 0.86), alpha=0.3)

                #Customize the plot
                plt.xlabel('depth (um)')
                plt.ylabel('Refractive index')
                plt.title('Structure profile')
                # plt.xlim(middle-middle/2, middle+middle/2)
                plt.ylim(0, 3)

                # Save the plot
                plt.savefig('structure_profile.png')

                ###############################################################################################################


                # Simulate the cavity
                aux = simulateQ(1, 0, lambda_cav, 0.01, 4000, N_top, N_bot ,Ncav, False)
                # aux = simulateQandeta(1, structure, 0, lambda_cav, 0.01, 100, Lcav, name_plot=folder_name)
                # aux = simulateQandeta(1, structure, 0, lambda_cav, 0.001, 100, Lcav, name_plot=folder_name)

                df = df.append({"lambda_cav":lambda_cav, "Lcav":Lcav, "N_top":N_top, "N_bot":N_bot, "Q_fact":aux["Qfact"], "peak":aux["peak"]}, ignore_index=True)

                finish_time = time.time()



                # Write aux to a txt file
                # with open(f'{folder_name}/data.txt', 'w') as file:
                #     file.write("R: " + str(aux["R"]))
                #     file.write('\n')
                #     file.write("lambda: " + str(aux["wavelengths"]))
                #     file.write('\n')
                #     file.write("linewidth: " + str(aux["linewidth"]))
                #     file.write('\n')
                #     file.write("Q: " + str(aux["Qfact"]))
                #     file.write('\n')
                #     file.write("absorption: " + str(aux["absorption"]))

                #     file.write("\nInitial parameters:\n")
                #     file.write(f"lambda_cav: {lambda_cav}\n")
                #     file.write(f"N_top: {N_top}\n")
                #     file.write(f"N_bot: {N_bot}\n")
                #     file.write(f"Lcav: {Lcav}\n")
                print(f'\nTime elapsed: {finish_time - start_time} seconds')
                it += 1
                print(f"Simulation finished for lambda_cav: {lambda_cav}, N_top: {N_top}, N_bot: {N_bot}, Lcav: {Lcav}, Q = {aux['Qfact']},it = {it}/{tot}")
                plt.close('all')

print(df)
df.to_csv("database.txt")
