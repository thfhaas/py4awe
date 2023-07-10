'''
File: viz_func.py
Author: Thomas Haas (Ghent University)
Date: 29/07/2022
Description: Visualization routines for awebox
'''

# ================================= #

# Imports
import os
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
from math import ceil, floor
import importlib
import matplotlib.animation as animation
import pickle
from matplotlib.patches import Patch

###from py4awes import loadawes
###importlib.reload(loadawes)

#===========================================================#
# '''
# Using Latex typesetting in plots
# '''
# import matplotlib
# matplotlib.rcParams['text.usetex'] = False
# matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
# ax.xaxis.get_major_formatter()._usetex = False
# ax.yaxis.get_major_formatter()._usetex = False
#===========================================================#

#======================================================================================================================#

def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v/256 for v in value]

def get_continuous_cmap(hex_list, float_list=None):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list.

        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.

        Returns
        ----------
        colour map'''

    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0, 1, len(rgb_list)))

    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mpl.colors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp

def create_figure():

    # Create figure
    s = 0.9
    fw = s*8.267
    fh = s*11.692
    fig = plt.figure(figsize=(fw, fh))

    # Create axes
    dw = 0.12
    dh = 0.08
    aw = 0.20
    ah = 0.24
    ax = []
    for j in range(3):
        for i in range(3):
            ax.append(fig.add_axes([(i+1)*dw + i*aw, (3-j)*dh + (2-j)*ah, aw, ah]))

    # ax.append(fig.add_axes([0.12, 0.08, 0.2, 0.24]))
    # ax.append(fig.add_axes([0.44, 0.08, 0.2, 0.24]))
    # ax.append(fig.add_axes([0.76, 0.08, 0.2, 0.24]))
    # #
    # ax.append(fig.add_axes([0.12, 0.40, 0.2, 0.24]))
    # ax.append(fig.add_axes([0.44, 0.40, 0.2, 0.24]))
    # ax.append(fig.add_axes([0.76, 0.40, 0.2, 0.24]))
    # #
    # ax.append(fig.add_axes([0.12, 0.72, 0.2, 0.24]))
    # ax.append(fig.add_axes([0.44, 0.72, 0.2, 0.24]))
    # ax.append(fig.add_axes([0.76, 0.72, 0.2, 0.24]))

    # Configure axes
    for ax_i in ax:
        ax_i.tick_params(axis='both', which='major', direction='in', labelsize=12)
        ax_i.grid(axis='both')

    # Add annotation
    import string
    for ax_i, a in zip(ax, list(string.ascii_lowercase[:len(ax)])):
            ax_i.annotate('('+a+')', xycoords='axes fraction', xy=(0.02, 0.90), ha='left', va='bottom', fontsize=12)

    return fig, ax

def plot_3d(awes):

    fig = plt.figure(figsize=(6.0, 6.0))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    fig.subplots_adjust(bottom=0.05, top=0.95, left=0.05, right=0.9)

    # Plot tether
    N = 20
    for m in range(N):
        k = m*int(len(awes['time'])/N)
        ax.plot([0.0, awes['x_q10_0'][k]], [0.0, awes['x_q10_1'][k]], [0.0, awes['x_q10_2'][k]], color='lightgray', linewidth=0.8)

    # Plot trajectory
    ax.plot(awes['x_q10_0'], awes['x_q10_1'], awes['x_q10_2'])

    P_inst = np.array(awes['outputs_performance_p_current_0'][:-1])
    dt = np.array(awes['time'][1:])-np.array(awes['time'][:-1])
    P_ave = np.sum(P_inst*dt)/awes['time'][-1]

    # Layout
    ax.set_xlabel(r'$x \, [\mathrm{m}]$', fontsize=12)
    ax.set_ylabel(r'$y \, [\mathrm{m}]$', fontsize=12)
    ax.set_zlabel(r'$z \, [\mathrm{m}]$', fontsize=12)

    # fs = 12. #10.
    # ax.set_xlim([0.0, 800.0])
    # ax.set_ylim([-400.0, 400.0])
    # ax.set_zlim([0.0, 800.0])
    # ax.view_init(elev=10., azim=-70.)
    # xticks = np.linspace(0.0, 800.0, 5)
    # yticks = np.linspace(-400., 400., 5)
    # zticks = np.linspace(0.0, 800.0, 5)
    # ax.set_xticks(xticks)
    # ax.set_yticks(yticks)
    # ax.set_zticks(zticks)
    # ax.tick_params(axis='both', which='major', labelsize=fs)
    # ax.legend(loc=1, ncol=2, bbox_to_anchor=(1.32, 1.0))
    # ax.legend(loc=1, ncol=2, bbox_to_anchor=(0.95, 1.0))
    # ax.legend(loc='upper right', ncol=2, bbox_to_anchor=(1.0, 0.9), fontsize=fs)
    # ax.legend(loc=1, bbox_to_anchor=(1.0, 0.9), fontsize=15)
    # ax.legend(ncol=2, loc='center right', bbox_to_anchor=(1.0, 0.9), fontsize=fs)
    # if len(letter) == 1:
    #     ax.annotate(s='('+letter[0]+')', xy=(0.02, 0.90), xycoords='figure fraction', fontsize=fs, va='center')

    return fig

def plot_motion(awes):

    # Create figure
    fig, ax = create_figure()

    # Add data
    for i in range(3):
        omega_i = (180.0 / np.pi) * np.array(awes['x_omega10_' + str(i)])
        ax[0+i].plot(awes['time'], awes['x_q10_' + str(i)])
        ax[3+i].plot(awes['time'], awes['x_dq10_' + str(i)])
        ax[6+i].plot(awes['time'], omega_i)

    # Add label
    import string
    for i, k in zip(range(3), list(string.ascii_lowercase[-3:])):
        ax[0+i].set_ylabel(r'$'+k+'\,[\mathrm{m}]$', fontsize=12)
        ax[3+i].set_ylabel(r'$\dot{'+k+'}\,[\mathrm{m}/\mathrm{s}]$', fontsize=12)
        ax[6+i].set_ylabel(r'$\omega_{'+k+'}\,[\deg/\mathrm{s}]$', fontsize=12)
    for i in range(len(ax)):
        ax[i].set_xlabel(r'$t\,[\mathrm{s}]$', fontsize=12)

    return fig

def plot_attitude(awes):

    # Create figure
    fig, ax = create_figure()

    # If DCM data is missing:
    N = len(awes['time'])
    if np.max(awes['x_r10_3']) <= 1.0e-6:
        # > Convert Euler angles to DCM
        rot = np.zeros((N, 9))
        for n in range(N):
            euler = []
            for fieldname in ['x_r10_0', 'x_r10_1', 'x_r10_2']:
                euler.append(awes[fieldname][n])
            rot[n, :] = np.squeeze(np.reshape(R.from_euler('xyz', euler, degrees=False).as_matrix(), (9, 1), order='F'))
        # > Overwrite DCM entries with computed data
        for k in range(9):
            fieldname = 'x_r10_'+str(k)
            awes[fieldname] = rot[:,k]

    # Add data
    for k in range(len(ax)):
            ax[k].plot(awes['time'], awes['x_r10_' + str(k)])

    # Add label
    for k in range(len(ax)):
        i = k%3+1
        j = int(k/3)+1
        ax[k].set_xlabel(r'$t\,[\mathrm{s}]$', fontsize=12)
        ax[k].set_ylabel(r'$R_{'+str(i)+str(j)+'}$', fontsize=12)

    return fig

def plot_controls(awes):

    # Create figure
    fig, ax = create_figure()

    # Add data
    ax[0].plot(awes['time'], awes['x_l_t_0'])
    ax[1].plot(awes['time'], awes['x_dl_t_0'])
    ax[2].step(awes['time'], awes['u_ddl_t_0'])
    for i in range(3):
        delta_i = (180.0/np.pi)*np.array(awes['x_delta10_' + str(i)])
        ddelta_i = (180.0/np.pi)*np.array(awes['u_ddelta10_' + str(i)])
        ax[3+i].plot(awes['time'], delta_i)
        ax[6+i].step(awes['time'], ddelta_i)

    # Add label
    ax[0].set_ylabel(r'$\ell\,[\mathrm{m}]$', fontsize=12)
    ax[1].set_ylabel(r'$\dot{\ell}\,[\mathrm{m}/\mathrm{s}]$', fontsize=12)
    ax[2].set_ylabel(r'$\ddot{\ell}\,[\mathrm{m}/\mathrm{s}^2]$', fontsize=12)
    import string
    for i, k in zip(range(3), ['e','a','r']):
        ax[3 + i].set_ylabel(r'$\delta_{' + k + '}\,[\deg/\mathrm{s}]$', fontsize=12)
        ax[6 + i].set_ylabel(r'$\dot{\delta_{' + k + '}}\,[\deg/\mathrm{s}^2]$', fontsize=12)
    for i in range(len(ax)):
        ax[i].set_xlabel(r'$t\,[\mathrm{s}]$', fontsize=12)

    return fig

def plot_aero_perf(awes):

    # Create figure
    fig, ax = create_figure()

    # Add data
    N = len(awes['time'])
    ddq = np.linalg.norm([awes['outputs_xdot_from_var_ddq10_' + '{:d}'.format(i)] for i in range(3)], axis=0)
    T = np.array(awes['z_lambda10_0'])*np.array(awes['x_l_t_0'])
    ax[0].plot(awes['time'], awes['outputs_environment_windspeed1_0'])
    ax[1].plot(awes['time'], np.zeros(N))
    ax[2].step(awes['time'], np.zeros(N))
    ax[3].plot(awes['time'], awes['outputs_aerodynamics_airspeed1_0'])
    ax[4].plot(awes['time'], awes['outputs_aerodynamics_alpha_deg1_0'])
    ax[5].plot(awes['time'], awes['outputs_aerodynamics_beta_deg1_0'])
    ax[6].plot(awes['time'], ddq)
    ax[7].step(awes['time'], 1.0e-3*T)
    ax[8].plot(awes['time'], 1.0e-6*np.array(awes['outputs_performance_p_current_0']))

    # Add label
    import string
    for i, k in zip(range(3), list(string.ascii_lowercase[-3:])):
        ax[0+i].set_ylabel(r'$V_{'+k+'}\,[\mathrm{m}/\mathrm{s}]$', fontsize=12)
    ax[3].set_ylabel(r'$||v_a||\,[\mathrm{m}/\mathrm{s}]$', fontsize=12)
    ax[4].set_ylabel(r'$\alpha\,[\deg]$', fontsize=12)
    ax[5].set_ylabel(r'$\beta\,[\deg]$', fontsize=12)
    ax[6].set_ylabel(r'$||\ddot{q}||\,[\mathrm{m}/\mathrm{s}^2]$', fontsize=12)
    ax[7].set_ylabel(r'$T\,[\mathrm{kN}]$', fontsize=12)
    ax[8].set_ylabel(r'$P\,[\mathrm{MW}]$', fontsize=12)
    for i in range(len(ax)):
        ax[i].set_xlabel(r'$t\,[\mathrm{s}]$', fontsize=12)

    return fig

def plot_aero_forces(awes):

    # Create figure
    fig, ax = create_figure()

    # Add data
    N = len(awes['time'])
    FL = np.linalg.norm([awes['outputs_aerodynamics_f_lift_earth1_' + '{:d}'.format(i)] for i in range(3)], axis=0)
    FD = np.linalg.norm([awes['outputs_aerodynamics_f_drag_earth1_' + '{:d}'.format(i)] for i in range(3)], axis=0)
    FT = np.linalg.norm([awes['outputs_tether_aero_multi_upper1_' + '{:d}'.format(i)] for i in range(3)], axis=0)
    for i, F in zip(range(3), [FL, FD, FT]):
        ax[0+i].plot(awes['time'], 1.0e-3*F)
        ax[3+i].plot(awes['time'], 1.0e-3*np.array(awes['outputs_aerodynamics_f_aero_body1_' + '{:d}'.format(i)]))
        ax[6+i].plot(awes['time'], 1.0e-3*np.array(awes['outputs_aerodynamics_m_aero_body1_' + '{:d}'.format(i)]))

    # Add label
    import string
    for i, v, k, p in zip(range(3), ['L','D','T'], list(string.ascii_lowercase[-3:]), list(string.ascii_lowercase[11:14])):
        ax[0+i].set_ylabel(r'$||F_{'+v+'}||\,[\mathrm{kN}]$', fontsize=12)
        ax[3+i].set_ylabel(r'$F_{'+k+'}\,[\mathrm{kN}]$', fontsize=12)
        ax[6+i].set_ylabel(r'$M_{'+p+'}\,[\mathrm{kNm}]$', fontsize=12)
    for i in range(len(ax)):
        ax[i].set_xlabel(r'$t\,[\mathrm{s}]$', fontsize=12)

    return fig

def plot_aero_rot(awes):

    # Create figure
    fig, ax = create_figure()

    # Compute Euler angles from DCM
    N = len(awes['time'])
    if np.max(awes['x_r10_3']) >= 1.0e-3:
        euler_angles = np.zeros((N, 3))
        for n in range(N):
            rot = []
            for fieldname in ['x_r10_' + str(i) for i in range(9)]:
                rot.append(awes[fieldname][n])
            euler_angles[n, :] = R.from_matrix(np.reshape(rot, (3, 3), order='F')).as_euler('xyz', degrees=False)

    # Add data
    for i in range(3):
        if np.max(awes['x_r10_3']) >= 1.0e-3:
            euler_i = (180.0 / np.pi) * euler_angles[:,i]
        else:
            euler_i = (180.0 / np.pi) * np.array(awes['x_r10_' + str(i)])
        omega_i = (180.0 / np.pi) * np.array(awes['x_omega10_' + str(i)])
        ax[0+i].plot(awes['time'], 1.0e-3*np.array(awes['outputs_aerodynamics_m_aero_body1_' + '{:d}'.format(i)]))
        ax[3+i].plot(awes['time'], omega_i)
        ax[6+i].plot(awes['time'], euler_i)

    # Add label
    import string
    for i, k, p in zip(range(3), list(string.ascii_lowercase[-3:]), list(string.ascii_lowercase[11:14])):
        ax[0 + i].set_ylabel(r'$M_{' + p + '}\,[\mathrm{kNm}]$', fontsize=12)
        ax[3 + i].set_ylabel(r'$\omega_{' + k + '}\,[\deg/\mathrm{s}]$', fontsize=12)
    ax[6].set_ylabel(r'$\phi\,[\deg]$', fontsize=12)
    ax[7].set_ylabel(r'$\theta\,[\deg]$', fontsize=12)
    ax[8].set_ylabel(r'$\psi\,[\deg]$', fontsize=12)
    for i in range(len(ax)):
        ax[i].set_xlabel(r'$t\,[\mathrm{s}]$', fontsize=12)

    return fig




# from stl import mesh
# from mpl_toolkits import mplot3d
# from matplotlib import pyplot
# from matplotlib.colors import LightSource
#
# # Create a new plot
# figure = pyplot.figure()
# axes = mplot3d.Axes3D(figure)
#
# # Load the STL files and add the vectors to the plot
# fname = '/cfdfile2/data/fm/thomash/Devs/ParaView/megawes_aircraft/aircraft_megawes_scaled.stl'
#
# stlmesh = mesh.Mesh.from_file(fname)
# import math
# stlmesh = stlmesh.rotate([0.0, 0.5, 0.0], math.radians(90))
#
# polymesh = mplot3d.art3d.Poly3DCollection(stlmesh.vectors)
#
# # Create light source
# ls = LightSource(azdeg=200,altdeg=65)
#
# # Darkest shadowed surface, in rgba
# dk = np.array([0.2, 0.0, 0.0, 1])
# # Brightest lit surface, in rgba
# lt = np.array([0.7, 0.7, 1.0, 1])
# # Interpolate between the two, based on face normal
# shade = lambda s: (lt - dk) * s + dk
#
# # Set face colors
# sns = ls.shade_normals(stlmesh.get_unit_normals(), fraction=1.0)
# rgba = np.array([shade(s) for s in sns])
# polymesh.set_facecolor(rgba)
#
# obj = axes.add_collection3d(polymesh)
#
# # Auto scale to the mesh size
# # scale = stlmesh.points.flatten(-1)
# # axes.auto_scale_xyz(scale, scale, scale)
#
#
#
# # Adjust limits of axes to fill the mesh, but keep 1:1:1 aspect ratio
# pts = stlmesh.points.reshape(-1, 3)
# ptp = max(np.ptp(pts, 0)) / 2
# ctrs = [(min(pts[:, i]) + max(pts[:, i])) / 2 for i in range(3)]
# lims = [[ctrs[i] - ptp, ctrs[i] + ptp] for i in range(3)]
# axes.auto_scale_xyz(*lims)
#
