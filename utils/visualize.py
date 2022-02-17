import torch

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20,10)

def plot_real_data(trainset):
    # v_params = {"beta":[],"gamma":[],"delta":[]}
    d_sir_values = {"S":[],"I":[],"R":[],"D":[]}
    for i in range(len(trainset)) :

        d_sir_values["S"].append(trainset[i][0][0].detach().numpy())
        d_sir_values["I"].append(trainset[i][0][1].detach().numpy())
        d_sir_values["R"].append(trainset[i][0][2].detach().numpy())
        d_sir_values["D"].append(trainset[i][0][3].detach().numpy())

    v_x = [i for i in range(len(d_sir_values["S"]))]

    fig, ax = plt.subplots(4)
    fig.suptitle('Real Data')

    ax[0].plot(v_x, d_sir_values["S"] , label = "S")
    ax[0].legend()
    ax[1].plot(v_x, d_sir_values["I"] , label = "I")
    ax[1].legend()
    ax[2].plot(v_x, d_sir_values["R"] , label = "R")
    ax[2].legend()
    ax[3].plot(v_x, d_sir_values["D"] , label = "D")
    ax[3].legend()

    return

