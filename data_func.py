'''
File: data_func.py
Author: Thomas Haas (Ghent University)
Date: 29/07/2022
Description: Data processing routines for awebox
'''

# Imports
import csv
import math as m
import numpy as np
from scipy.spatial.transform import Rotation as R

# -------------------------- AWEBOX import -------------------------- #
def csv2dict(fname):
    '''
    Import CSV outputs from awebox to Python dictionary
    '''

    # read csv file
    with open(fname, 'r') as f:
        reader = csv.DictReader(f)

        # get fieldnames from DictReader object and store in list
        headers = reader.fieldnames

        # store data in columns
        columns = {}
        for row in reader:
            for fieldname in headers:
                val = row.get(fieldname).strip('[]')
                if val == '':
                    val = '0.0'
                columns.setdefault(fieldname, []).append(float(val))

    # add periodicity
    for fieldname in headers:
        columns.setdefault(fieldname, []).insert(0, columns[fieldname][-1])
    columns['time'][0] = 0.0

    return columns

def dict2t(data):
    '''
    Extract time as numpy array from Python dictionary
    '''
    t = np.array(data['time'])
    return t

def dict2x(data, return_t=False, print_keys=False):
    '''
    Extract states as numpy array from Python dictionary
    '''
    # List of states (dict keys)
    raw_keys_states = [idx for idx in list(data.keys()) if idx.lower().startswith('x')]
    keys_states = list()
    keys_states.extend(['x_q10_0', 'x_q10_1', 'x_q10_2'])
    keys_states.extend(['x_dq10_0', 'x_dq10_1', 'x_dq10_2'])
    keys_states.extend(['x_omega10_0', 'x_omega10_1', 'x_omega10_2'])
    keys_states.extend(
        ['x_r10_0', 'x_r10_1', 'x_r10_2', 'x_r10_3', 'x_r10_4', 'x_r10_5', 'x_r10_6', 'x_r10_7', 'x_r10_8'])
    keys_states.extend(['x_delta10_0', 'x_delta10_1', 'x_delta10_2'])
    keys_states.extend(['x_l_t_0', 'x_dl_t_0'])
    if print_keys:
        for key in keys_states:
            print(key)

    # Time
    t = np.array(data['time'])

    # States
    x = np.empty((len(t), len(keys_states)))
    for k, key in zip(range(len(keys_states)), keys_states):
        x[:,k] = list(data[key])

    # Return
    if return_t:
        return t, x
    else:
        return x

def dict2u(data, return_t=False, print_keys=False):
    '''
    Extract controls as numpy array from Python dictionary
    '''
    # List of controls (dict keys)
    raw_keys_states = [idx for idx in list(data.keys()) if idx.lower().startswith('u')]
    keys_controls = ['u_ddl_t_0', 'u_ddelta10_0', 'u_ddelta10_1', 'u_ddelta10_2']
    if print_keys:
        for key in keys_controls:
            print(key)

    # Time
    t = np.array(data['time'])

    # Controls
    u = np.empty((len(t), len(keys_controls)))
    for k, key in zip(range(len(keys_controls)), keys_controls):
        u[:,k] = list(data[key])

    # Return
    if return_t:
        return t, u
    else:
        return u

# -------------------------- Rotation matrices -------------------------- #
def Rx(theta):
    '''
    Rotation about x-axis
    '''
    return np.array([[ 1, 0           , 0           ],
                      [ 0, m.cos(theta),-m.sin(theta)],
                      [ 0, m.sin(theta), m.cos(theta)]])

def Ry(theta):
    '''
    Rotation about y-axis
    '''
    return np.array([[ m.cos(theta), 0, m.sin(theta)],
                      [ 0           , 1, 0           ],
                      [-m.sin(theta), 0, m.cos(theta)]])

def Rz(theta):
    '''
    Rotation about z-axis
    '''
    return np.array([[ m.cos(theta), -m.sin(theta), 0 ],
                      [ m.sin(theta), m.cos(theta) , 0 ],
                      [ 0           , 0            , 1 ]])

# -------------------------- MegAWES stability derivatives -------------------------- #
def get_aero_stab_derivs(model='AVL'):
    '''
    Return aerodynamic stability derivatives of MegAWES aircraft
    Computed by Jolan Wauters with AVL. Implemented as such in AWEBOX.
    '''
    stab_derivs = {}

    # Force coefficients (MegAWES)
    # TODO: Verify deltaa has no contribution to forces?
    stab_derivs['CX'] = {}
    stab_derivs['CX']['0'] = [-0.054978] # 1
    stab_derivs['CX']['alpha'] = [0.95827, 5.6521] # alpha,alpha2
    stab_derivs['CX']['q'] = [-0.62364, 3.9554, 0.76064] # 1,alpha,alpha2
    stab_derivs['CX']['deltae'] = [-0.031786, 0.2491, 0.076416] # 1,alpha,alpha2

    stab_derivs['CY'] = {}
    stab_derivs['CY']['beta'] = [-0.25065, -0.077264, -0.25836] # 1,alpha,alpha2
    stab_derivs['CY']['p'] = [-0.11406, -0.52471, 0.029861]
    stab_derivs['CY']['r'] = [0.094734, 0.029343, -0.053453]
    stab_derivs['CY']['deltae'] = [-0.0072147, 0.034068, 0.0032198] #TODO: Should this not be deltaa ?
    stab_derivs['CY']['deltar'] = [0.22266, 0.0099775, -0.21563]

    stab_derivs['CZ'] = {}
    stab_derivs['CZ']['0'] = [-1.2669] # 1
    stab_derivs['CZ']['alpha'] = [-6.3358, 0.17935] # alpha,alpha2
    stab_derivs['CZ']['q'] = [-8.5019, -1.0139, 3.5051] # 1,alpha,alpha2
    stab_derivs['CZ']['deltae'] = [-0.56729, -0.0090401, 0.53686]

    # Moment coefficients (MegAWES)
    stab_derivs['Cl'] = {}
    stab_derivs['Cl']['beta'] = [0.033447, 0.29221, -0.008944]
    stab_derivs['Cl']['p'] = [-0.62105, -0.013963, 0.19592]
    stab_derivs['Cl']['r'] = [0.29112, 0.74473, -0.09468]
    stab_derivs['Cl']['deltaa'] = [-0.22285, 0.072142, 0.20538]
    stab_derivs['Cl']['deltar'] = [0, 0, 0]

    stab_derivs['Cm'] = {}
    stab_derivs['Cm']['0'] = [0.046452] # 1
    stab_derivs['Cm']['alpha'] = [0.19869, 0.34447] # alpha,alpha2
    stab_derivs['Cm']['q'] = [-9.81, 0.013751, 4.7861] # 1,alpha,alpha2
    stab_derivs['Cm']['deltae'] = [-1.435, -0.0079607, 1.3813]

    stab_derivs['Cn'] = {}
    stab_derivs['Cn']['beta'] = [0.043461, 0.0014151, -0.034616] # 1,alpha,alpha2
    stab_derivs['Cn']['p'] = [-0.10036, -1.0669, -0.016814]
    stab_derivs['Cn']['r'] = [-0.043558, 0.030031, -0.12246]
    stab_derivs['Cn']['deltaa'] = [0.0066928, -0.086616, 0.0005028]
    stab_derivs['Cn']['deltar'] = [-0.04988, -0.00096109, 0.04822]

    return stab_derivs

# -------------------------- MegAWES aerodynamic model -------------------------- #
def C(C_a,alpha):
    '''
    Evaluate aerodynamic coefficient
    C = a0 + a1*alpha + a2*alpha^2

    NOTE TO NP: You did np.matmul(C_a,np.array([alpha**2,alpha,1]).T)
    '''
    return np.matmul(C_a,np.array([1,alpha,alpha**2]).T)

def compute_aero_force_coefs(b, c, alpha, beta, omega, Va, delta):
    '''
    Compute aerodynamic force coefficients of main wing of MegAWES aircraft
    Inputs: Aerodynamic quantities b, c, alpha, beta, p, q, r, Va, deltaa, deltae, deltar
    Outputs: Aerodynamic force coefficient
    '''
    #TODO: Is it necessary to have the coefficients as (3,1) arrays and matrices instead of (1,3) arrays ?

    # angular velocities
    p = omega[0]
    q = omega[1]
    r = omega[2]

    # surface control deflections
    deltaa = delta[0]
    deltae = delta[1]
    deltar = delta[2]

    # Aerodynamic stability derivatives of MegAWES aircraft
    stab_derivs = get_aero_stab_derivs('AVL')

    # alpha contribution (main wing)
    CX_0_a_MW = np.concatenate((stab_derivs['CX']['0'], stab_derivs['CX']['alpha']))
    CY_0_a_MW = np.array([0, 0, 0])
    CZ_0_a_MW = np.concatenate((stab_derivs['CZ']['0'], stab_derivs['CZ']['alpha']))
    # CF_0 = np.array([[C(CX_0_a_MW, alpha), C(CY_0_a_MW, alpha), C(CZ_0_a_MW, alpha)]]).T
    CF_0 = np.array([C(CX_0_a_MW, alpha), C(CY_0_a_MW, alpha), C(CZ_0_a_MW, alpha)])

    # beta contribution
    CX_B_a_MW = np.array([0, 0, 0])
    CY_B_a_MW = stab_derivs['CY']['beta']
    CZ_B_a_MW = np.array([0, 0, 0])
    # CF_B = np.array([[C(CX_B_a_MW, alpha), C(CY_B_a_MW, alpha), C(CZ_B_a_MW, alpha)]]).T
    CF_B = np.array([C(CX_B_a_MW, alpha), C(CY_B_a_MW, alpha), C(CZ_B_a_MW, alpha)])
    CF_beta = CF_B*beta

    # roll contribution (p)
    CX_p_a_MW = np.array([0, 0, 0])
    CY_p_a_MW = stab_derivs['CY']['p']
    CZ_p_a_MW = np.array([0, 0, 0])

    # pitch contribution (q)
    CX_q_a_MW = stab_derivs['CX']['q']
    CY_q_a_MW = np.array([0, 0, 0])
    CZ_q_a_MW = stab_derivs['CZ']['q']

    # yaw contribution (r)
    #TODO: stab_derivs['CY']['deltar'] = np.array([0.22266, 0.0099775, -0.21563]) was first used
    #TODO: stab_derivs['CY']['r'] = np.array([0.094734, 0.029343, -0.053453]) is probably correct
    CX_r_a_MW = np.array([0, 0, 0])
    CY_r_a_MW = stab_derivs['CY']['r']
    CZ_r_a_MW = np.array([0, 0, 0])

    # Contribution of angular rates p, q, r #TODO: better name ?
    # pqr_norm = np.array([[b * p / (2 * Va), c * q / (2 * Va), b * r / (2 * Va)]]).T
    # CF_rot = np.matrix([[C(CX_p_a_MW, alpha), C(CX_q_a_MW, alpha), C(CX_r_a_MW, alpha)],
    #                     [C(CY_p_a_MW, alpha), C(CY_q_a_MW, alpha), C(CY_r_a_MW, alpha)],
    #                     [C(CZ_p_a_MW, alpha), C(CZ_q_a_MW, alpha), C(CZ_r_a_MW, alpha)]])
    # CF_pqr = CF_rot * pqr_norm
    pqr_norm = np.array([b * p / (2 * Va), c * q / (2 * Va), b * r / (2 * Va)])
    CF_rot = np.array([[C(CX_p_a_MW, alpha), C(CX_q_a_MW, alpha), C(CX_r_a_MW, alpha)],
                       [C(CY_p_a_MW, alpha), C(CY_q_a_MW, alpha), C(CY_r_a_MW, alpha)],
                       [C(CZ_p_a_MW, alpha), C(CZ_q_a_MW, alpha), C(CZ_r_a_MW, alpha)]])
    CF_pqr = np.matmul(CF_rot, pqr_norm)

    # aileron contribution
    #TODO: Verify why deltaa has no contributions
    CX_deltaa_a_MW = np.array([0, 0, 0])
    CY_deltaa_a_MW = np.array([0, 0, 0]) #TODO: This should probably not be zero
    CZ_deltaa_a_MW = np.array([0, 0, 0])
    # CF_da = np.array([[C(CX_deltaa_a_MW, alpha), C(CY_deltaa_a_MW, alpha), C(CZ_deltaa_a_MW, alpha)]]).T
    CF_da = np.array([C(CX_deltaa_a_MW, alpha), C(CY_deltaa_a_MW, alpha), C(CZ_deltaa_a_MW, alpha)])
    CF_deltaa = CF_da*deltaa

    # elevator contribution
    CX_deltae_a_MW = stab_derivs['CX']['deltae']
    CY_deltae_a_MW = stab_derivs['CY']['deltae'] #TODO: This should maybe be np.array([0, 0, 0])
    CZ_deltae_a_MW = stab_derivs['CZ']['deltae']
    # CF_de = np.array([[C(CX_deltae_a_MW, alpha), C(CY_deltae_a_MW, alpha), C(CZ_deltae_a_MW, alpha)]]).T
    CF_de = np.array([C(CX_deltae_a_MW, alpha), C(CY_deltae_a_MW, alpha), C(CZ_deltae_a_MW, alpha)])
    CF_deltae = CF_de*deltae

    # rudder contribution
    CX_deltar_a_MW = np.array([0, 0, 0])
    CY_deltar_a_MW = stab_derivs['CY']['deltar']
    CZ_deltar_a_MW = np.array([0, 0, 0])
    # CF_dr = np.array([[C(CX_deltar_a_MW, alpha), C(CY_deltar_a_MW, alpha), C(CZ_deltar_a_MW, alpha)]]).T
    CF_dr = np.array([C(CX_deltar_a_MW, alpha), C(CY_deltar_a_MW, alpha), C(CZ_deltar_a_MW, alpha)])
    CF_deltar = CF_dr*deltar

    # contribution of control surfaces
    CF_delta = CF_deltaa + CF_deltae + CF_deltar

    # Total aerodynamic coefficient
    CF_tot = CF_0 + CF_beta + CF_pqr + CF_delta

    return CF_tot, CF_0, CF_beta, CF_pqr, CF_delta

def compute_aero_moment_coefs(b, c, alpha, beta, omega, Va, delta):
    '''
    Compute aerodynamic force coefficients of main wing of MegAWES aircraft
    Inputs: Aerodynamic quantities b, c, alpha, beta, p, q, r, Va, deltaa, deltae, deltar
    Outputs: Aerodynamic force coefficient
    '''
    #TODO: Is it necessary to have the coefficients as (3,1) arrays and matrices instead of (1,3) arrays ?

    # angular velocities
    p = omega[0]
    q = omega[1]
    r = omega[2]

    # surface control deflections
    deltaa = delta[0]
    deltae = delta[1]
    deltar = delta[2]

    # Aerodynamic stability derivatives of MegAWES aircraft
    stab_derivs = get_aero_stab_derivs('AVL')

    # alpha contribution (main wing)
    Cl_0_a_MW = np.array([0, 0, 0])
    Cm_0_a_MW = np.concatenate((stab_derivs['Cm']['0'], stab_derivs['Cm']['alpha']))
    Cn_0_a_MW = np.array([0, 0, 0])

    CM_0 = np.array([C(Cl_0_a_MW, alpha), C(Cm_0_a_MW, alpha), C(Cn_0_a_MW, alpha)])

    # beta contribution
    Cl_B_a_MW = stab_derivs['Cl']['beta']
    Cm_B_a_MW = np.array([0, 0, 0])
    Cn_B_a_MW = stab_derivs['Cn']['beta']

    CM_B = np.array([C(Cl_B_a_MW, alpha), C(Cm_B_a_MW, alpha), C(Cn_B_a_MW, alpha)])
    CM_beta = CM_B*beta

    # roll contribution (p)
    Cl_p_a_MW = stab_derivs['Cl']['p']
    Cm_p_a_MW = np.array([0, 0, 0])
    Cn_p_a_MW = stab_derivs['Cn']['p']

    # pitch contribution (q)
    Cl_q_a_MW = np.array([0, 0, 0])
    Cm_q_a_MW = stab_derivs['Cm']['q']
    Cn_q_a_MW = np.array([0, 0, 0])

    # yaw contribution (r)
    Cl_r_a_MW = stab_derivs['Cl']['r']
    Cm_r_a_MW = np.array([0, 0, 0])
    Cn_r_a_MW = stab_derivs['Cn']['r']

    # Contribution of angular rates p, q, r #TODO: better name ?

    pqr_norm = np.array([b * p / (2 * Va), c * q / (2 * Va), b * r / (2 * Va)])
    CM_rot = np.array([[C(Cl_p_a_MW, alpha), C(Cl_q_a_MW, alpha), C(Cl_r_a_MW, alpha)],
                       [C(Cm_p_a_MW, alpha), C(Cm_q_a_MW, alpha), C(Cm_r_a_MW, alpha)],
                       [C(Cn_p_a_MW, alpha), C(Cn_q_a_MW, alpha), C(Cn_r_a_MW, alpha)]])
    CM_pqr = np.matmul(CM_rot, pqr_norm)

    # aileron contribution
    Cl_deltaa_a_MW = stab_derivs['Cl']['deltaa']
    Cm_deltaa_a_MW = np.array([0, 0, 0])
    Cn_deltaa_a_MW = stab_derivs['Cn']['deltaa']
    CM_da = np.array([C(Cl_deltaa_a_MW, alpha), C(Cm_deltaa_a_MW, alpha), C(Cn_deltaa_a_MW, alpha)])
    CM_deltaa = CM_da*deltaa

    # elevator contribution
    Cl_deltae_a_MW = np.array([0, 0, 0])
    Cm_deltae_a_MW = stab_derivs['Cm']['deltae'] 
    Cn_deltae_a_MW = np.array([0, 0, 0])
    CM_de = np.array([C(Cl_deltae_a_MW, alpha), C(Cm_deltae_a_MW, alpha), C(Cn_deltae_a_MW, alpha)])
    CM_deltae = CM_de*deltae

    # rudder contribution
    Cl_deltar_a_MW = stab_derivs['Cl']['deltar']
    Cm_deltar_a_MW = np.array([0, 0, 0])
    Cn_deltar_a_MW = stab_derivs['Cn']['deltar']
    CM_dr = np.array([C(Cl_deltar_a_MW, alpha), C(Cm_deltar_a_MW, alpha), C(Cn_deltar_a_MW, alpha)])
    CM_deltar = CM_dr*deltar

    # contribution of control surfaces
    CM_delta = CM_deltaa + CM_deltae + CM_deltar

    # Total aerodynamic coefficient
    CM_tot = CM_0 + CM_beta + CM_pqr + CM_delta

    return CM_tot, CM_0, CM_beta, CM_pqr, CM_delta

# -------------------------- AWEBOX aero model -------------------------- #
def compute_wind_speed(z, uref=10.0, zref=100.0, z0=0.0002, model = 'uniform', exp_ref = 0.15):
    '''
    compute logarithmic wind speed at specific height z
    '''
    if model == 'log_wind':

        # mathematically: it doesn't make a difference what the base of
        # these logarithms is, as long as they have the same base.
        # but, the values will be smaller in base 10 (since we're describing
        # altitude differences), which makes convergence nicer.
        # u = u_ref * np.log10(zz / z0_air) / np.log10(z_ref / z0_air)
        u = uref * np.log10(z / z0) / np.log10(zref / z0)

    elif model == 'power':
        u = uref * (z / zref) ** exp_ref

    elif model == 'uniform':
        u = uref

    else:
        raise ValueError('unsupported atmospheric option chosen: %s', model)

    return u

def compute_density(z, rho_ref=1.225, t_ref=288.15, gamma_air=6.5e-3, g=9.81, R_gas=287.053):
    '''
    compute density at specific height z
    '''
    return rho_ref * ((t_ref - gamma_air*z)/ t_ref) ** (g/gamma_air/R_gas - 1.0)

def compute_apparent_speed(q, dq, wind_model):
    '''
    compute apparent wind speed at specific height z
    '''
    Vw = np.array([compute_wind_speed(q[2], model = wind_model), 0, 0])
    Va = Vw - dq
    return Va, np.linalg.norm(Va)

def compute_DCM(r_ii):
    '''
    compute Direct Cosine Matrix (DCM)
    '''
    # AWEBOX outputs in Euler form
    if np.sum(np.abs(r_ii[3:])) <= 1e-3:
        euler_angles = r_ii[:3]
        R_rot = R.from_euler('xyz', euler_angles, degrees=False).as_matrix()
    # AWEBOX outputs in DCM form
    else:
        R_rot = np.reshape(r_ii, (3, 3), order='F')
    return R_rot

def compute_aero_angles(Va, R_rot):
    '''
    compute angle of attack and sideslip angle
    '''
    alpha = np.dot(R_rot[:,2], Va)/np.dot(R_rot[:,0], Va)
    beta = np.dot(R_rot[:,1], Va)/np.dot(R_rot[:,0], Va)
    return alpha, beta

# -------------------------- MegAWES aerodynamic forces and moments -------------------------- #
def compute_aero_forces(x, geom, wind_model):
    '''
    Compute aerodynamic forces of MegAWES aircraft
    '''

    # /!\ Malz aero model; give it in its own convention (=! awebox)
    R_awebox2malz = Ry(np.pi)
    # Compute Direct Cosine Matrix
    R_rot = compute_DCM(x[9:18])

    # Compute aerodynamic quantities from states (Va, alpha, beta) and air density
    Va, Va_norm = compute_apparent_speed(x[:3], x[3:6], wind_model)
    alpha, beta = compute_aero_angles(Va, R_rot)
    rho = compute_density(x[2])

    # Compute aerodynamic coefficients
    alpha, beta = compute_aero_angles(R_awebox2malz @ Va, R_awebox2malz @ R_rot @ R_awebox2malz.T) # Malz iner 2 Malz body = Malz 2 awebow + awebox inert 2 awebox body + awebox 2 Malz

    # /!\ Malz aero model; give it in its awn convention (=! awebox)
    R_awebox2malz = Rx(np.pi)@Rz(np.pi)

    coefs = compute_aero_force_coefs(geom.b, geom.c, alpha, beta, R_awebox2malz@x[6:9], Va_norm, x[18:21])
    CF = coefs[0] # Total force

    # Compute aerodynamic forces
    F = 0.5*rho*CF*geom.S*Va_norm**2

    return R_awebox2malz.T @ F, R_awebox2malz.T @ CF, alpha, beta, Va

# -------------------------- MegAWES aerodynamic forces and moments -------------------------- #
def compute_aero_moments(x, geom, wind_model):
    '''
    Compute aerodynamic forces of MegAWES aircraft
    '''

    # /!\ Malz aero model; give it in its awn convention (=! awebox)
    R_awebox2malz = Ry(np.pi)
    # Compute Direct Cosine Matrix
    R_rot = compute_DCM(x[9:18])

    # Compute aerodynamic quantities from states (Va, alpha, beta) and air density
    Va, Va_norm = compute_apparent_speed(x[:3], x[3:6], wind_model)
    alpha, beta = compute_aero_angles(Va, R_rot)
    alpha, beta = compute_aero_angles(R_awebox2malz @ Va, R_awebox2malz @ R_rot @ R_awebox2malz.T) # Malz iner 2 Malz body = Malz 2 awebow + awebox inert 2 awebox body + awebox 2 Malz
    rho = compute_density(x[2])

    # Compute aerodynamic coefficients

    # /!\ Malz aero model; give it in its awn convention (=! awebox)
    R_awebox2malz = Rx(np.pi)@Rz(np.pi)

    coefs = compute_aero_moment_coefs(geom.b, geom.c, alpha, beta, R_awebox2malz@x[6:9], Va_norm, x[18:21])
    CM = coefs[0] # Total force
    l = np.array([geom.b, geom.c, geom.b])
    # Compute aerodynamic forces
    M = 0.5*rho*CM*l*geom.S*Va_norm**2

    return R_awebox2malz.T @ M, R_awebox2malz.T @ CM

# -------------------------- MegAWES aircraft instance -------------------------- #
class Geometry:
    '''
    Geometry of MegAWES aircraft: span, aspect ratio, surface area and equivalent chord length
    '''
    def __init__(self, b=42.47, AR=12., S=150.45):
        self.b = b
        self.AR = AR
        self.S = S
        self.c = self.S/self.b

class Aerodynamics:
    '''
    Instantaneous aerodynamics of MegAWES aircraft
    '''
    def __init__(self, x, geom, wind_model):
        F, CF, alpha, beta, Va = compute_aero_forces(x, geom, wind_model)
        M, CM = compute_aero_moments(x, geom, wind_model)

        self.forces = F
        self.moments = M
        self.aeroCoefs = np.concatenate((CF, CM))
        self.alpha = alpha
        self.beta = beta
        self.Va = Va


class MegAWES:
    def __init__(self, t, x, wind_model = 'uniform'):
        self.t = t
        self.x = x
        self.geom = Geometry()
        self.aero = Aerodynamics(x, self.geom, wind_model)

    # # Returns wing and tail forces in
    # return F_aero, F_wing, F_tail
    #
    # R2 = Rz(-beta) * Ry(alpha)  # transformation body to aerodynamic
    #
    # CFa_tot = R2 * CF_tot
