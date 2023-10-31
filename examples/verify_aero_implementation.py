#!usr/bin/python3
'''
Verification of AWEbox implementations of aerodynamic model:
> Sign conventions of sideslip angle and control surface deflections.
> Force and moment computations in "aero" and "body" frames.
Author: Thomas Haas (Ghent University)
Date: 31/10/2023

Frames:
- "aero" frame is defined as forward-right-down (aircraft convention).
- "body" frame is defined as backward-right-up (AWEbox convention).
- Transformation from one frame to another is a 180deg rotation about local y-axis.

Sideslip angle:
- Positive sideslip angle generates negative side force in "aero" frame (negative in "body" frame).
- Positive sideslip angle generates positive restoring yaw moment in "aero" frame (negative in "body" frame).
- The sideslip angle outputted by AWEbox is incorrect and should be multiplied by -1.

Control surfaces:
- Positive aileron deflection (left up/right down) generates a negative roll moment in the "aero" frame.
- Positive elevator deflection (TE down) generates a negative vertical force and a negative pitching moment in the "aero" frame.
- Positive rudder deflection (towards left wingtip) generates a positive side force and a negative yaw moment in the "aero" frame.
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
filename1 = "./megawes_uniform_1loop_results_old_beta.csv"
filename2 = "./megawes_uniform_1loop_results_new_beta.csv"
output_folder = "./examples/" 

# -------------------------- Get AWEBOX data -------------------------- #
'''
Example 1: Get AWEBOX data
'''

# Retrieve data Dict from AWEBOX
awes1 = fun.csv2dict(filename1)
awes2 = fun.csv2dict(filename2)

# -------------------------- Get output keys -------------------------- #

# List of aerodynamics outputs
#out = [i for i in list(awes.keys()) if i.startswith('outputs_aerodynamics_air')]
for key in list(awes1.keys()):
    if key.startswith('outputs_aerodynamics'):
        print(key)

# List of environment outputs
for key in list(awes1.keys()):
    if key.startswith('outputs_environment'):
        print(key)

# -------------------------- Compare AWEbox outputs -------------------------- #

# Plot 3D flight path
viz.plot_3d(awes1)
fig = plt.gcf()
fig.set_size_inches(8., 8.)
ax = fig.get_axes()[0]
ax.plot(awes2['x_q10_0'], awes2['x_q10_1'], awes2['x_q10_2'])
l = ax.get_lines()
l[-2].set_color('tab:blue')
l[-1].set_color('tab:green')
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend([l[-2],l[-1]], [r"Old $\beta$ implementation", r"New $\beta$ implementation"], loc=1, fontsize=12)
ax.set_title("Comparison of side-slip angle implementation", fontsize=12)
fig.savefig(output_folder+"comparison_traj3d.png")

# Plot side-slip angle
fig, ax = plt.subplots(figsize=(8.,8.))
ax.plot(awes1['time'], awes1['outputs_aerodynamics_beta_deg1_0'], color='tab:blue', label=r"Old $\beta$ implementation")
ax.plot(awes2['time'], awes2['outputs_aerodynamics_beta_deg1_0'], color='tab:green', label=r"New $\beta$ implementation")
ax.plot(awes2['time'], (-1)*np.array(awes2['outputs_aerodynamics_beta_deg1_0']), ':', color='tab:gray', label=r'(-1)x New $\beta$')
ax.set_xlim([0, 15])
ax.set_xticks(np.linspace(0., 15., 6))
ax.set_xlabel(r"Time $t$", fontsize=12)
ax.set_ylim([-11, 11])
ax.set_yticks(np.linspace(-10., 10., 5))
ax.set_ylabel(r"Side-slip angle $\beta$", fontsize=12)
ax.grid()
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend(loc=2, fontsize=12)
ax.set_title("Comparison of side-slip angle implementation", fontsize=12)
fig.savefig(output_folder+"comparison_beta.png")

# -------------------------- Reconstruct AWEbox outputs in body frame -------------------------- #

# Loop results of old/new implementations
for awes, implementation in zip([awes1, awes2], ["old", "new"]):

    # Retrieve states
    t_ref, x_ref = fun.dict2x(awes, return_t=True)
    aero_frame = "body"

    # Compute forces and moments
    F = np.empty((len(t_ref), 3))
    M = np.empty((len(t_ref), 3))
    beta = np.empty((len(t_ref), 1))
    for k, (t, x) in enumerate(zip(t_ref, x_ref)):
        MegAWES = fun.MegAWES(t, x, wind_model = 'uniform')
        F[k,:] = MegAWES.aero.F[aero_frame]
        M[k,:] = MegAWES.aero.M[aero_frame]
        beta[k] = (180./np.pi)*MegAWES.aero.beta

    # Reconstruct aero from AWEbox outputs
    b = 42.47
    S = 150.45
    c = S/b
    alpha = np.array(awes['outputs_aerodynamics_alpha1_0'])
    beta_out = -np.array(awes['outputs_aerodynamics_beta1_0'])
    Va = np.array(awes['outputs_aerodynamics_airspeed1_0'])
    rho = np.array(awes['outputs_environment_density1_0'])
    Fy = []
    Ml = []
    Mn = []
    for k in range(len(awes['time'])):
        omega = np.array([j*awes['x_omega10_'+str(i)][k] for i, j in zip(range(3),[-1,1,-1])])
        delta = np.array([j*awes['x_delta10_'+str(i)][k] for i, j in zip(range(3),[1,1,1])])
        Va_norm = Va[k]
        CF_tot, CF_0, CF_beta, CF_pqr, CF_delta = fun.compute_aero_force_coefs(b, c, alpha[k], beta_out[k], omega, Va_norm, delta)
        CM_tot, CM_0, CM_beta, CM_pqr, CM_delta = fun.compute_aero_moment_coefs(b, c, alpha[k], beta_out[k], omega, Va_norm, delta)
        Fy.append(0.5 * rho[k] * S * CF_tot[1] * Va_norm ** 2)
        Ml.append(0.5 * rho[k] * S * b * -CM_tot[0] * Va_norm ** 2) # Minus sign because of aero-body transformation
        Mn.append(0.5 * rho[k] * S * b * -CM_tot[2] * Va_norm ** 2) # Minus sign because of aero-body transformation

    # Plot side-slip angle
    fig, ax = plt.subplots(figsize=(8.,8.))
    ax.plot(awes['time'], awes['outputs_aerodynamics_beta_deg1_0'], 'k-', label='AWEbox output (in body frame)')
    ax.plot(t_ref, beta, color='tab:green', label='Computed from states (in aero frame)')
    ax.set_xlim([0, 15])
    ax.set_xticks(np.linspace(0., 15., 6))
    ax.set_xlabel(r"Time $t$", fontsize=12)
    ax.set_ylim([-11, 11])
    ax.set_yticks(np.linspace(-10., 10., 5))
    ax.set_ylabel(r"Side-slip angle $\beta$", fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid()
    ax.legend(loc=2, fontsize=12)
    ax.set_title("Comparison of sideslip angle outputs", fontsize=12)
    fig.savefig(output_folder + "comparison_beta_"+implementation+"_implementation.png")

    # Plot forces
    fig, ax = plt.subplots(figsize=(8.,8.), nrows=3)
    plt.subplots_adjust(hspace=0.4)
    ax[0].plot(awes['time'], awes['outputs_aerodynamics_f_aero_'+aero_frame+'1_0'], 'k-', label='AWEbox output')
    ax[0].plot(awes['time'], F[:,0], color='tab:green', label='Computed from states')
    ax[1].plot(awes['time'], awes['outputs_aerodynamics_f_aero_'+aero_frame+'1_1'], 'k-', label='AWEbox output')
    ax[1].plot(awes['time'], Fy, '-', color='tab:gray', label=r'from AWEbox using $-\beta$')
    ax[1].plot(awes['time'], F[:,1], color='tab:green', label='Computed from states')
    ax[2].plot(awes['time'], awes['outputs_aerodynamics_f_aero_'+aero_frame+'1_2'], 'k-', label='AWEbox output')
    ax[2].plot(awes['time'], F[:,2], color='tab:green', label='Computed from states')
    ax[1].legend(loc=2, fontsize=12)
    ax[0].set_ylim([-5e4,5e4])
    ax[0].set_yticks(np.linspace(-5e4,5e4,5))
    ax[1].set_ylim([-5e4,5e4])
    ax[1].set_yticks(np.linspace(-5e4, 5e4, 5))
    ax[2].set_ylim([0,1e6])
    ax[2].set_yticks(np.linspace(0,1e6,6))
    for k in range(3):
        ax[k].grid()
        ax[k].tick_params(axis='both', which='major', labelsize=12)
        ax[k].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax[k].set_xlim([0., 15.])
        ax[k].set_xticks(np.linspace(0., 15., 6))
        ax[k].set_xlabel(r'Time $t$', fontsize=12)
        ax[k].set_ylabel(r'Force $F_'+str(k)+'$', fontsize=12)
    ax[0].set_title("Comparison of forces in body frame", fontsize=12)
    fig.savefig(output_folder + "comparison_forces_"+implementation+"_implementation.png")

    # Plot moments
    fig, ax = plt.subplots(figsize=(8.,8.), nrows=3)
    plt.subplots_adjust(hspace=0.4)
    ax[0].plot(awes['time'], awes['outputs_aerodynamics_m_aero_'+aero_frame+'1_0'], 'k-', label='AWEbox output')
    ax[0].plot(awes['time'], Ml, color='tab:gray', label=r'from AWEbox using $-\beta$')
    ax[0].plot(awes['time'], M[:,0], color='tab:green', label='Computed from states')
    ax[1].plot(awes['time'], awes['outputs_aerodynamics_m_aero_'+aero_frame+'1_1'], 'k-', label='AWEbox output')
    ax[1].plot(awes['time'], M[:,1], color='tab:green', label='Computed from states')
    ax[2].plot(awes['time'], awes['outputs_aerodynamics_m_aero_'+aero_frame+'1_2'], 'k-', label='AWEbox output')
    ax[2].plot(awes['time'], Mn, '-', color='tab:gray', label=r'from AWEbox using $-\beta$')
    ax[2].plot(awes['time'], M[:,2], color='tab:green', label='Computed from states')
    ax[0].legend(loc=2, fontsize=12)
    ax[2].legend(loc=2, fontsize=12)
    ax[0].set_ylim([-5e5, 5e5])
    ax[0].set_yticks(np.linspace(-5e5, 5e5, 5))
    ax[1].set_ylim([-5e4, 5e4])
    ax[1].set_yticks(np.linspace(-5e4, 5e4, 5))
    ax[2].set_ylim([-5e5, 5e5])
    ax[2].set_yticks(np.linspace(-5e5, 5e5, 5))
    for k in range(3):
        ax[k].grid()
        ax[k].tick_params(axis='both', which='major', labelsize=12)
        ax[k].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax[k].set_xlim([0., 15.])
        ax[k].set_xticks(np.linspace(0., 15., 6))
        ax[k].set_xlabel(r'Time $t$', fontsize=12)
        ax[k].set_ylabel(r'Moment $M_'+str(k)+'$', fontsize=12)
    ax[0].set_title("Comparison of moments in body frame", fontsize=12)
    fig.savefig(output_folder + "comparison_moments_"+implementation+"_implementation.png")

# -------------------------- Sign convention of control surface deflections in aero frame -------------------------- #

# 1. Plot total coefficients vs. AOA without other contributions

# Retrieve coefficients for range of AOA
CM_tot = []
CF_tot = []
alpha = np.linspace(-20, 10, 301)
for aoa in alpha:
    CM = fun.compute_aero_moment_coefs(b, c, aoa*(np.pi/180.), 0., 3*[0.], 80., [0., 0., 0.])
    CF = fun.compute_aero_force_coefs(b, c, aoa*(np.pi/180.), 0., 3*[0.], 80., [0., 0., 0.])
    CM_tot.append(CM[0])
    CF_tot.append(CF[0])

# Plot forces vs AOA
fig, ax = plt.subplots(figsize=(8., 8.), nrows=3)
plt.subplots_adjust(hspace=0.4)
for k in range(3):
    ax[k].plot(alpha, [CF_tot[i][k] for i in range(len(CF_tot))], '-', color='tab:blue')
ax[0].set_ylim([-.5, .5])
ax[1].set_ylim([-.5, .5])
ax[2].set_ylim([-3., 3.])
ax[0].set_yticks(np.linspace(-.5, .5, 5))
ax[1].set_yticks(np.linspace(-.5, .5, 5))
ax[2].set_yticks(np.linspace(-3., 3., 7))
for k in range(3):
    ax[k].grid()
    ax[k].tick_params(axis='both', which='major', labelsize=12)
    ax[k].set_xlim([-20., 10.])
    ax[k].set_xticks(np.linspace(-20., 10., 7))
    ax[k].set_xlabel(r'angle-of-attack $\alpha$ [deg]', fontsize=12)
    ax[k].set_ylabel(r'coeff $C_{F,'+str(k)+'}$', fontsize=12)
ax[0].set_title("Force coeff in aero frame (AOA contribution only)", fontsize=12)
fig.savefig(output_folder + "force_coeffs_aero_frame.png")

# Plot moments vs AOA
fig, ax = plt.subplots(figsize=(8., 8.), nrows=3)
plt.subplots_adjust(hspace=0.4)
for k in range(3):
    ax[k].plot(alpha, [CM_tot[i][k] for i in range(len(CM_tot))], '-', color='tab:blue')
ax[0].set_ylim([-.05, .05])
ax[1].set_ylim([0, 0.1])
ax[2].set_ylim([-.05, .05])
ax[0].set_yticks(np.linspace(-.05, .05, 5))
ax[1].set_yticks(np.linspace(.0, 0.1, 6))
ax[2].set_yticks(np.linspace(-.05, .05, 5))
for k in range(3):
    ax[k].grid()
    ax[k].tick_params(axis='both', which='major', labelsize=12)
    ax[k].set_xlim([-20., 10.])
    ax[k].set_xticks(np.linspace(-20., 10., 7))
    ax[k].set_xlabel(r'angle-of-attack $\alpha$ [deg]', fontsize=12)
    ax[k].set_ylabel(r'coeff $C_{M,' + str(k) + '}$', fontsize=12)
ax[0].set_title("Moment coeff in aero frame (AOA contribution only)", fontsize=12)
fig.savefig(output_folder + "moment_coeffs_aero_frame.png")

# 2. Plot coefficients vs. surface deflections

# Specific AOA close to zero-lift AOA
alpha = (-11.40)*(np.pi/180.0)

# Range of surface deflections
deltaa = np.linspace(-10, 10, 11)
deltae = np.linspace(-10, 10, 11)
deltar = np.linspace(-10, 10, 11)

# Initializations for force coeffs
CF_tot_deltaa = []
CF_tot_deltae = []
CF_tot_deltar = []
CF_delta_deltaa = []
CF_delta_deltae = []
CF_delta_deltar = []

# Initializations for moment coeffs
CM_tot_deltaa = []
CM_tot_deltae = []
CM_tot_deltar = []
CM_delta_deltaa = []
CM_delta_deltae = []
CM_delta_deltar = []

# Compute coefficients
for da, de, dr in zip(deltaa, deltae, deltar):

    # Aileron
    CM = fun.compute_aero_moment_coefs(b, c, alpha, 0., 3*[0.], 80., [da*(np.pi/180.), 0., 0.])
    CF = fun.compute_aero_force_coefs(b, c, alpha, 0., 3*[0.], 80., [da*(np.pi/180.), 0., 0.])
    CM_tot_deltaa.append(CM[0])
    CM_delta_deltaa.append(CM[4])
    CF_tot_deltaa.append(CF[0])
    CF_delta_deltaa.append(CF[4])

    # Elevator
    CM = fun.compute_aero_moment_coefs(b, c, alpha, 0., 3*[0.], 80., [0., de*(np.pi/180.), 0.])
    CF = fun.compute_aero_force_coefs(b, c, alpha, 0., 3*[0.], 80., [0., de*(np.pi/180.), 0.])
    CM_tot_deltae.append(CM[0])
    CM_delta_deltae.append(CM[4])
    CF_tot_deltae.append(CF[0])
    CF_delta_deltae.append(CF[4])

    # Elevator
    CM = fun.compute_aero_moment_coefs(b, c, alpha, 0., 3*[0.], 80., [0., 0., de*(np.pi/180.)])
    CF = fun.compute_aero_force_coefs(b, c, alpha, 0., 3*[0.], 80., [0., 0., de*(np.pi/180.)])
    CM_tot_deltar.append(CM[0])
    CM_delta_deltar.append(CM[4])
    CF_tot_deltar.append(CF[0])
    CF_delta_deltar.append(CF[4])

# Plot force coefficients
fig, ax = plt.subplots(figsize=(8., 8.), nrows=3)
plt.subplots_adjust(hspace=0.4)
for k in range(3):
    ax[k].plot(deltaa, [CF_tot_deltaa[i][k] for i in range(len(CF_tot_deltaa))], '-.', color='tab:gray', label=r'$\delta_a$')
    ax[k].plot(deltae, [CF_tot_deltae[i][k] for i in range(len(CF_tot_deltae))], ':', color='tab:gray', label=r'$\delta_e$')
    ax[k].plot(deltar, [CF_tot_deltar[i][k] for i in range(len(CF_tot_deltar))], '--', color='tab:gray', label=r'$\delta_r$')
    ax[k].plot(deltaa, [CF_delta_deltaa[i][k] for i in range(len(CF_delta_deltaa))], '-', color='tab:blue', label=r'$\delta_a$')
    ax[k].plot(deltae, [CF_delta_deltae[i][k] for i in range(len(CF_delta_deltae))], '-', color='tab:green', label=r'$\delta_e$')
    ax[k].plot(deltar, [CF_delta_deltar[i][k] for i in range(len(CF_delta_deltar))], '-', color='tab:purple', label=r'$\delta_r$')
ax[0].legend(ncol=2)
ax[0].set_ylim([-.05, .05])
ax[1].set_ylim([-.05, .05])
ax[2].set_ylim([-.1, .1])
ax[0].set_yticks(np.linspace(-.05, .05, 5))
ax[1].set_yticks(np.linspace(-.05, .05, 5))
ax[2].set_yticks(np.linspace(-.1, 0.1, 5))
for k in range(3):
    ax[k].grid()
    ax[k].tick_params(axis='both', which='major', labelsize=12)
    ax[k].set_xlim([-10., 10.])
    ax[k].set_xticks(np.linspace(-10., 10., 11))
    ax[k].set_xlabel(r'deflection $\delta_i$ [deg]', fontsize=12)
    ax[k].set_ylabel(r'coeff $C_{F,' + str(k) + '}$', fontsize=12)
ax[0].set_title("Force coefficients in aero frame due to deflections", fontsize=12)
fig.savefig(output_folder + "force_coeffs_aero_frame_delta.png")

# Plot moment coefficients
fig, ax = plt.subplots(figsize=(8., 8.), nrows=3)
plt.subplots_adjust(hspace=0.4)
for k in range(3):
    ax[k].plot(deltaa, [CM_tot_deltaa[i][k] for i in range(len(CM_tot_deltaa))], '-.', color='tab:gray', label=r'$\delta_a$')
    ax[k].plot(deltae, [CM_tot_deltae[i][k] for i in range(len(CM_tot_deltae))], ':', color='tab:gray', label=r'$\delta_e$')
    ax[k].plot(deltar, [CM_tot_deltar[i][k] for i in range(len(CM_tot_deltar))], '--', color='tab:gray', label=r'$\delta_r$')
    ax[k].plot(deltaa, [CM_delta_deltaa[i][k] for i in range(len(CM_delta_deltaa))], '-', color='tab:blue', label=r'$\delta_a$')
    ax[k].plot(deltae, [CM_delta_deltae[i][k] for i in range(len(CM_delta_deltae))], '-', color='tab:green', label=r'$\delta_e$')
    ax[k].plot(deltar, [CM_delta_deltar[i][k] for i in range(len(CM_delta_deltar))], '-', color='tab:purple', label=r'$\delta_r$')
ax[0].legend(ncol=2)
ax[0].set_ylim([-.05, .05])
ax[1].set_ylim([-.5, .5])
ax[2].set_ylim([-.01, .01])
ax[0].set_yticks(np.linspace(-.05, .05, 5))
ax[1].set_yticks(np.linspace(-.5, .5, 5))
ax[2].set_yticks(np.linspace(-.01, 0.01, 5))
for k in range(3):
    ax[k].grid()
    ax[k].tick_params(axis='both', which='major', labelsize=12)
    ax[k].set_xlim([-10., 10.])
    ax[k].set_xticks(np.linspace(-10., 10., 11))
    ax[k].set_xlabel(r'deflection $\delta_i$ [deg]', fontsize=12)
    ax[k].set_ylabel(r'coeff $C_{M,' + str(k) + '}$', fontsize=12)
ax[0].set_title("Moment coefficients in aero frame due to deflections", fontsize=12)
fig.savefig(output_folder + "moment_coeffs_aero_frame_delta.png")

# -------------------------- Contributions of control surface deflections in AWEbox -------------------------- #

# Rudder contribution: Fy and Mn
# Elevator contribution: Mm (and Fz)
# Aileron contribution: Ml

# Retrieve states
t_ref, x_ref = fun.dict2x(awes2, return_t=True)
aero_frame = "body"

# Aircraft dimensions
b = 42.47
S = 150.45
c = S/b

# Initialize contributions
Fw = []
Fa = []
Fe = []
Fr = []
Ft = []
Mw = []
Ma = []
Me = []
Mr = []
Mt = []

# Retrieve quantities
alpha = np.array(awes['outputs_aerodynamics_alpha1_0'])
beta = np.array(awes['outputs_aerodynamics_beta1_0'])
Va = np.array(awes['outputs_aerodynamics_airspeed1_0'])
rho = np.array(awes['outputs_environment_density1_0'])

# Loop through time
for k in range(len(awes['time'])):

    # Required states
    omega = np.array([j*awes['x_omega10_'+str(i)][k] for i, j in zip(range(3),[-1,1,-1])])
    delta = np.array([j*awes['x_delta10_'+str(i)][k] for i, j in zip(range(3),[1,1,1])])
    Va_norm = Va[k]

    # Wing contribution (delta = 0)
    CFw = fun.compute_aero_force_coefs(b, c, alpha[k], beta_out[k], omega, Va_norm, 3*[0])
    CMw = fun.compute_aero_moment_coefs(b, c, alpha[k], beta_out[k], omega, Va_norm, 3*[0])

    # Aileron contribution
    CFa = fun.compute_aero_force_coefs(b, c, alpha[k], beta_out[k], omega, Va_norm, [delta[0], 0., 0.])
    CMa = fun.compute_aero_moment_coefs(b, c, alpha[k], beta_out[k], omega, Va_norm, [delta[0], 0., 0.])

    # Elevator contribution
    CFe = fun.compute_aero_force_coefs(b, c, alpha[k], beta_out[k], omega, Va_norm, [0., delta[1], 0.])
    CMe = fun.compute_aero_moment_coefs(b, c, alpha[k], beta_out[k], omega, Va_norm, [0., delta[1], 0.])

    # Rudder contribution
    CFr = fun.compute_aero_force_coefs(b, c, alpha[k], beta_out[k], omega, Va_norm, [0., 0., delta[2]])
    CMr = fun.compute_aero_moment_coefs(b, c, alpha[k], beta_out[k], omega, Va_norm, [0., 0., delta[2]])

    # Forces expressed in aero frame
    Fw.append(0.5 * rho[k] * S * CFw[0] * Va_norm ** 2)
    Fa.append(0.5 * rho[k] * S * CFa[4] * Va_norm ** 2)
    Fe.append(0.5 * rho[k] * S * CFe[4] * Va_norm ** 2)
    Fr.append(0.5 * rho[k] * S * CFr[4] * Va_norm ** 2)
    Ft.append(Fw[-1] + Fa[-1] + Fe[-1] + Fr[-1])

    # Moments expressed in aero frame
    scale = np.array([b, c, b])
    Mw.append(0.5 * rho[k] * S * scale * CMw[0] * Va_norm ** 2)
    Ma.append(0.5 * rho[k] * S * scale * CMa[4] * Va_norm ** 2)
    Me.append(0.5 * rho[k] * S * scale * CMe[4] * Va_norm ** 2)
    Mr.append(0.5 * rho[k] * S * scale * CMr[4] * Va_norm ** 2)
    Mt.append(Mw[-1] + Ma[-1] + Me[-1] + Mr[-1])

# Plot control surface deflections
fig, ax = plt.subplots(figsize=(8.,8.))
ax.plot(t_ref, (180./np.pi)*x_ref[:,18], color='tab:blue', label=r'aileron $\delta_a$')
ax.plot(t_ref, (180./np.pi)*x_ref[:,19], color='tab:green', label=r'elevator $\delta_e$')
ax.plot(t_ref, (180./np.pi)*x_ref[:,20], color='tab:purple', label=r'rudder $\delta_r$')
ax.set_xlim([0, 15])
ax.set_xticks(np.linspace(0., 15., 6))
ax.set_xlabel(r"Time $t$", fontsize=12)
ax.set_ylim([-20, 20])
ax.set_yticks(np.linspace(-20., 20., 9))
ax.set_ylabel(r"Deflection $\delta_i$", fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.grid()
ax.legend(loc=1, fontsize=12)
ax.set_title("Control surface deflections", fontsize=12)
fig.savefig(output_folder + "control_surface_deflections.png")

# Plot force contributions
fig, ax = plt.subplots(figsize=(8., 8.), nrows=3)
plt.subplots_adjust(hspace=0.4)
for k, sgn in zip(range(3), [-1, 1, -1]):
    ax[k].plot(t_ref, awes['outputs_aerodynamics_f_aero_body1_'+str(k)], color='tab:gray', label='AWEbox')
    ax[k].plot(t_ref, sgn*np.array([Ft[i][k] for i in range(len(t_ref))]), color='black', label='total')
    ax[k].plot(t_ref, sgn*np.array([Fw[i][k] for i in range(len(t_ref))]), color='tab:red', label='wing')
    ax[k].plot(t_ref, sgn*np.array([Fa[i][k] for i in range(len(t_ref))]), color='tab:blue', label='aileron')
    ax[k].plot(t_ref, sgn*np.array([Fe[i][k] for i in range(len(t_ref))]), color='tab:green', label='elevator')
    ax[k].plot(t_ref, sgn*np.array([Fr[i][k] for i in range(len(t_ref))]), color='tab:purple', label='rudder')
ax[0].legend(loc=2, ncol=2, fontsize=12)
ax[0].set_ylim([-4e4,4e4])
ax[1].set_ylim([-3e4,3e4])
ax[2].set_ylim([-2e5,1e6])
ax[0].set_yticks(np.linspace(-4e4,4e4,9))
ax[1].set_yticks(np.linspace(-3e4, 3e4, 7))
ax[2].set_yticks(np.linspace(-2e5,1e6,7))
for k in range(3):
    ax[k].grid()
    ax[k].tick_params(axis='both', which='major', labelsize=12)
    ax[k].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax[k].set_xlim([0., 15.])
    ax[k].set_xticks(np.linspace(0., 15., 6))
    ax[k].set_xlabel(r'Time $t$', fontsize=12)
    ax[k].set_ylabel(r'Force $F_'+str(k)+'$', fontsize=12)
ax[0].set_title("Force contributions in body frame", fontsize=12)
fig.savefig(output_folder + "comparison_forces_contributions.png")

# Plot moment contributions
fig, ax = plt.subplots(figsize=(8., 8.), nrows=3)
plt.subplots_adjust(hspace=0.4)
for k, sgn in zip(range(3), [-1, 1, -1]):
    ax[k].plot(t_ref, awes['outputs_aerodynamics_m_aero_body1_'+str(k)], color='tab:gray', label='AWEbox')
    ax[k].plot(t_ref, sgn*np.array([Mt[i][k] for i in range(len(t_ref))]), color='black', label='total')
    ax[k].plot(t_ref, sgn*np.array([Mw[i][k] for i in range(len(t_ref))]), color='tab:red', label='wing')
    ax[k].plot(t_ref, sgn*np.array([Ma[i][k] for i in range(len(t_ref))]), color='tab:blue', label='aileron')
    ax[k].plot(t_ref, sgn*np.array([Me[i][k] for i in range(len(t_ref))]), color='tab:green', label='elevator')
    ax[k].plot(t_ref, sgn*np.array([Mr[i][k] for i in range(len(t_ref))]), color='tab:purple', label='rudder')
ax[0].legend(loc=2, ncol=2, fontsize=12)
ax[0].set_ylim([-1.2e6, 1.2e6])
ax[1].set_ylim([-1.2e5, 1.2e5])
ax[2].set_ylim([-3e5, 3e5])
ax[0].set_yticks(np.linspace(-1.2e6, 1.2e6, 7))
ax[1].set_yticks(np.linspace(-1.2e5, 1.2e5, 7))
ax[2].set_yticks(np.linspace(-3e5, 3e5, 7))
for k in range(3):
    ax[k].grid()
    ax[k].tick_params(axis='both', which='major', labelsize=12)
    ax[k].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax[k].set_xlim([0., 15.])
    ax[k].set_xticks(np.linspace(0., 15., 6))
    ax[k].set_xlabel(r'Time $t$', fontsize=12)
    ax[k].set_ylabel(r'Moment $M_'+str(k)+'$', fontsize=12)
ax[0].set_title("Moment contributions in body frame", fontsize=12)
fig.savefig(output_folder + "comparison_moments_contributions.png")

# -------------------------- End -------------------------- #
