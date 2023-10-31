#!usr/bin/python3
'''
Example file for the py4awe package
Authors: 
    Thomas Haas (Ghent University)
    Niels Pynaert (Ghent University)
    Jean-Baptiste Crismer (UC Louvain)
Date: 31/10/2023
'''
import os
import importlib
import numpy as np
from matplotlib import pyplot as plt
from py4awe import viz_func as viz
from py4awe import data_func as fun
importlib.reload(fun)
importlib.reload(viz)

# -------------------------- Initialization/User settings -------------------------- #
'''
User settings: Specify file name
'''
plt.close("all")
path = os.getcwd()
filename = "/cfdfile2/data/fm/thomash/Tools/py4awe/examples/megawes_uniform_1loop_results.csv"

# -------------------------- Get AWEbox data -------------------------- #
'''
Example 1: Get AWEbox data
'''

# Retrieve data Dict from AWEbox
awes = fun.csv2dict(filename)

# Plot 3D flight path
viz.plot_3d(awes)

# Extract reference time, states, and controls from Dict
# States: q = x[0:3], dq = x[3:6], R = x[6:15], omega = x[15:18], delta = x[18:21], l = x[21], dl = x[22]
# Controls: ddelta = u[0:3], ddl = u[-1]
t_ref = fun.dict2t(awes)
x_ref = fun.dict2x(awes)
u_ref = fun.dict2u(awes)

# -------------------------- Compute MegAWES aerodynamics at time instance -------------------------- #
'''
Example 2: Compute MegAWES aerodynamics at time instance
'''

# Random time in power cycle
from random import uniform
awes = fun.csv2dict(filename)
t_ref = fun.dict2t(awes)
idx = (np.abs(t_ref - uniform(0, t_ref.max()))).argmin()

# Aerodynamics at specific instance
t = t_ref[idx]
x = x_ref[idx,:]
MegAWES = fun.MegAWES(t, x, wind_model = 'uniform')
f_str = '[%s]' % ', '.join(['{:.3f}'.format(i) for i in 1e-3*MegAWES.aero.F['body']])
m_str = '[%s]' % ', '.join(['{:.3f}'.format(i) for i in 1e-3*MegAWES.aero.M['body']])
print('The aircraft span and AR are '+'{:.2f}'.format(MegAWES.geom.b)+' and '+'{:.2f}'.format(MegAWES.geom.AR)+'.')
print('At time '+'{:.2f}'.format(t)+', the forces and moments, expressed in body frame, are '+f_str+' kN and '+m_str+' kNm.')

# -------------------------- Reconstruct AWEbox forces and moments -------------------------- #
'''
Example 3: Reconstruct AWEBOX forces and moments
'''

# Retrieve size of dataset
awes = fun.csv2dict(filename)
t_ref, x_ref = fun.dict2x(awes, return_t=True)
N = len(t_ref)

# Initialize aerodynamic quantities
Fa = np.empty((N, 3))
Ma = np.empty((N, 3))
Fe = np.empty((N, 3))
Mb = np.empty((N, 3))
R_aero = np.empty((N, 9))
R_body = np.empty((N, 9))

# Reconstruct aerodynamics for each time instance
for k, t, x in zip(range(N), t_ref, x_ref):

    # MegAWES instance
    MegAWES = fun.MegAWES(t, x, wind_model='uniform')

    # Forces and moments in "aero" frame
    # Note: F, M are also directly available in "body" and "earth" frames.
    Fa[k,:] = MegAWES.aero.F['aero']
    Ma[k,:] = MegAWES.aero.M['aero']

    # Forces in inertial frame ("earth")
    R_aero[k,:] = np.reshape(MegAWES.aero.R_aero, (9,), order='F')
    Fe[k,:] = np.matmul(np.reshape(R_aero[k,:], (3,3), order='F'), MegAWES.aero.F['aero'])

    # Moments in body frame
    R_body[k,:] = np.reshape(MegAWES.aero.R_body, (9,), order='F')
    R_a2b = np.matmul(np.reshape(R_body[k, :], (3, 3), order='F').T, np.reshape(R_aero[k, :], (3, 3), order='F'))
    Mb[k,:] = np.matmul(R_a2b, MegAWES.aero.M['aero'])

# Plot forces and moments (with/out AWEbox reference, in aero/body/inertial frame)
fig1, ax1 = viz.plot_forces(t_ref, Fa)
ax1[0].legend(['computed'], loc=1)
#
fig2, ax2 = viz.plot_moments(t_ref, Mb, awes=awes, frame='body')
ax2[0].legend(['reference', 'computed'], loc=1)
#
fig3, ax3 = viz.plot_forces(t_ref, Fe, awes=awes, frame='earth')
for ax in ax3:
    lines = ax.get_lines()
    lines[0].set_linewidth(0.5)
    lines[-1].set_color('tab:green')
    ax.legend(['reference', 'computed'], loc=1)

# -------------------------- Compare with AWEBOX outputs -------------------------- #
'''
Example 4: Compare with awebox outputs
'''

# Retrieve size of dataset
awes = fun.csv2dict(filename)
t_ref, x_ref = fun.dict2x(awes, return_t=True)
N = len(t_ref)

# Initialize aerodynamic quantities
alpha = np.empty((N))
beta = np.empty((N))
Va = np.empty((N,3))

# Reconstruct aerodynamics for each time instance
for k, t, x in zip(range(N), t_ref, x_ref):
    MegAWES = fun.MegAWES(t, x, 'uniform')
    alpha[k] = MegAWES.aero.alpha
    beta[k] = MegAWES.aero.beta
    Va[k,:] = MegAWES.aero.Va

# Plot position
pos = x_ref[:,:3]
fig,ax = viz.plot_position(t_ref, pos, awes=awes)

# Plot flight speed
vel = x_ref[:,3:6]
fig,ax = viz.plot_velocity(t_ref, vel, awes=awes)

# Plot angular speed
omega = x_ref[:,15:18]
fig,ax = viz.plot_omega(t_ref, omega, awes=awes)

# Plot actuation
delta = x_ref[:,18:21]
fig,ax = viz.plot_actuation(t_ref, delta, awes=awes)

# Plot aero quantities
aero = np.concatenate((np.linalg.norm(Va, axis=1)[:,None], (180.0/np.pi)*alpha[:,None], (180.0/np.pi)*beta[:,None]), axis=1)
fig,ax = viz.plot_aero_quantities(t_ref, aero, awes=awes)
plt.show()
