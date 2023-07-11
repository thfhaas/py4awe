#!usr/bin/python3
'''
Example file for the py4awe package
Author: Thomas Haas, Ghent University
Date: 10/07/2023
'''
import os
import importlib
import numpy as np
from matplotlib import pyplot as plt
from awebox.py4awe import viz_func as viz
from awebox.py4awe import data_func as fun
importlib.reload(fun)

# -------------------------- Initialization/User settings -------------------------- #
'''
User settings: Specify file name
'''
plt.close("all")
path = os.getcwd()
filename = path+"/awebox/py4awe/examples/megAWES_outputs_1loop.csv"

# -------------------------- Get AWEBOX data -------------------------- #
# Retrieve data Dict from AWEBOX
awes = fun.csv2dict(filename)

# Plot 3D flight path
viz.plot_3d(awes)

# Extract time, states, and controls from Dict
t0 = fun.dict2t(awes)
x0 = fun.dict2x(awes)
u0 = fun.dict2u(awes)

# -------------------------- Recompute MegAWES aerodynamics -------------------------- #
#TODO: Fix aerodynamic model

# /!\ Malz model in forward, right, down body frame; Awebox model in backward, right, up body frame

# Compute aerodynamic forces at random time in power cycle
from random import uniform
idx = (np.abs(t0 - uniform(0, t0.max()))).argmin()
MegAWES = fun.MegAWES(t0[idx], x0[idx,:])
print(MegAWES.geom.b, MegAWES.geom.c)
print(MegAWES.aero.aeroCoefs, MegAWES.aero.forces)

# Compare computed forces and AWEBOX forces
N = len(t0)
F = np.empty((N, 3))
M = np.empty((N, 3))
for k, t, x in zip(range(N), t0, x0):
    MegAWES = fun.MegAWES(t, x)
    F[k,:] = MegAWES.aero.forces
    M[k,:] = MegAWES.aero.moments


# Plot forces
fig, ax =plt.subplots(figsize=(8,6), nrows=3)
for k in range(3):
    ax[k].plot(t0, F[:,k], label='computed')
    ax[k].plot(awes['time'], awes['outputs_aerodynamics_f_aero_body1_'+str(k)], label='awebox')
    ax[k].set_xlabel('time')
    ax[k].set_ylabel('F_'+str(k))
    ax[k].legend(loc=1)

# Plot moments
fig, ax =plt.subplots(figsize=(8,6), nrows=3)
for k in range(3):
    ax[k].plot(t0, M[:,k], label='computed')
    ax[k].plot(awes['time'], awes['outputs_aerodynamics_m_aero_body1_'+str(k)], label='awebox')
    ax[k].set_xlabel('time')
    ax[k].set_ylabel('M_'+str(k))
    ax[k].legend(loc=1)

plt.show()
