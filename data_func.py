'''
File: data_func.py
Authors: 
    Thomas Haas (Ghent University)
    Niels Pynaert (Ghent University)
    Jean-Baptiste Crismer (UC Louvain)
Date: 29/07/2022
Description: Data processing routines for awebox
'''

# Imports
import csv
import math as m
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys

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
    keys_states.extend(
        ['x_r10_0', 'x_r10_1', 'x_r10_2', 'x_r10_3', 'x_r10_4', 'x_r10_5', 'x_r10_6', 'x_r10_7', 'x_r10_8'])
    keys_states.extend(['x_omega10_0', 'x_omega10_1', 'x_omega10_2'])
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
    keys_controls = ['u_ddelta10_0', 'u_ddelta10_1', 'u_ddelta10_2', 'u_ddl_t_0']
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

# -------------------------- MegAWES aerodynamic coefficients -------------------------- #
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
    CF_0 = np.array([C(CX_0_a_MW, alpha), C(CY_0_a_MW, alpha), C(CZ_0_a_MW, alpha)])

    # beta contribution
    CX_B_a_MW = np.array([0, 0, 0])
    CY_B_a_MW = stab_derivs['CY']['beta']
    CZ_B_a_MW = np.array([0, 0, 0])
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
    CX_r_a_MW = np.array([0, 0, 0])
    CY_r_a_MW = stab_derivs['CY']['r']
    CZ_r_a_MW = np.array([0, 0, 0])

    # Contribution of angular rates p, q, r
    pqr_norm = np.array([b * p / (2 * Va), c * q / (2 * Va), b * r / (2 * Va)])
    CF_rot = np.array([[C(CX_p_a_MW, alpha), C(CX_q_a_MW, alpha), C(CX_r_a_MW, alpha)],
                       [C(CY_p_a_MW, alpha), C(CY_q_a_MW, alpha), C(CY_r_a_MW, alpha)],
                       [C(CZ_p_a_MW, alpha), C(CZ_q_a_MW, alpha), C(CZ_r_a_MW, alpha)]])
    CF_pqr = np.matmul(CF_rot, pqr_norm)

    # aileron contribution #TODO: Verify why deltaa has no contributions
    CX_deltaa_a_MW = np.array([0, 0, 0])
    CY_deltaa_a_MW = np.array([0, 0, 0]) #TODO: This should probably not be zero
    CZ_deltaa_a_MW = np.array([0, 0, 0])
    CF_da = np.array([C(CX_deltaa_a_MW, alpha), C(CY_deltaa_a_MW, alpha), C(CZ_deltaa_a_MW, alpha)])
    CF_deltaa = CF_da*deltaa

    # elevator contribution
    CX_deltae_a_MW = stab_derivs['CX']['deltae']
    CY_deltae_a_MW = stab_derivs['CY']['deltae'] #TODO: This should maybe be np.array([0, 0, 0])
    CZ_deltae_a_MW = stab_derivs['CZ']['deltae']
    CF_de = np.array([C(CX_deltae_a_MW, alpha), C(CY_deltae_a_MW, alpha), C(CZ_deltae_a_MW, alpha)])
    CF_deltae = CF_de*deltae

    # rudder contribution
    CX_deltar_a_MW = np.array([0, 0, 0])
    CY_deltar_a_MW = stab_derivs['CY']['deltar']
    CZ_deltar_a_MW = np.array([0, 0, 0])
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

    # Contribution of angular rates p, q, r
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

def retrieve_DCM(r_ii):
    '''
    compute Direct Cosine Matrix (DCM)
    '''
    # AWEBOX outputs in Euler form
    if np.sum(r_ii[3:]) <= 1e-3:
        euler_angles = r_ii[:3]
        R_rot = R.from_euler('xyz', euler_angles, degrees=False).as_matrix()
    # AWEBOX outputs in DCM form
    else:
        R_rot = np.reshape(r_ii, (3, 3), order='F')
    return R_rot

def compute_aero_angles(Va, frame):
    '''
    compute angle of attack and sideslip angle (small angle approximation tanTheta = Theta)
    Take Va in Malz's frame (fun input is in inertial frame)
    '''
    if frame=="aero":
        alpha = Va[2]/Va[0]
        beta = Va[1]/Va[0]
    else:
        sys.exit("Va needs to be given in the 'aero' frame. Exit.")
    return alpha, beta

# -------------------------- MegAWES aircraft dimensions -------------------------- #
class Geometry:
    '''
    Geometry of MegAWES aircraft: span, aspect ratio, surface area and equivalent chord length
    '''
    def __init__(self, b=42.47, AR=12., S=150.45):
        self.b = b
        self.AR = AR
        self.S = S
        self.c = self.S/self.b

# -------------------------- MegAWES aircraft aerodynamics -------------------------- #
class Aerodynamics:
    '''
    Instantaneous aerodynamics of MegAWES aircraft (expressed in awebox convention of body-frame)
    '''
    def __init__(self, x, geom, wind_model="uniform"):

        # ---------- Compute DCMs ---------- #

        # Retrieve DCM of body-fixed axis system ("body")  in AWEbox convention (backward-right-up)
        R_body = retrieve_DCM(x[6:15])
        self.R_body = R_body

        # Retrieve DCM of aerodynamics axis system ("aero") in Malz's convention (forward-right-down)
        R_aero = np.matmul(R_body, Ry(np.pi))
        self.R_aero = R_aero

        # ---------- Compute apparent wind speed ---------- #
        Va_earth, Va_norm = compute_apparent_speed(x[:3], x[3:6], wind_model=wind_model)
        Va_aero = np.matmul(R_aero.T, Va_earth)
        self.Va = Va_aero

        # ---------- Compute aerodynamic angles in the "aero" frame ---------- #
        alpha, beta = compute_aero_angles(Va_aero, frame="aero")
        self.alpha = alpha
        self.beta = beta

        # ---------- Compute air density ---------- #
        rho = compute_density(x[2])
        self.rho = rho

        # ---------- Compute forces and moments coefficients in the "aero" frame ---------- #
        omega_aero = np.matmul(Ry(np.pi), x[15:18])
        delta = x[18:21] # Ready to use in "aero" frame
        CF = compute_aero_force_coefs(geom.b, geom.c, alpha, beta, omega_aero, Va_norm, delta)
        CM = compute_aero_moment_coefs(geom.b, geom.c, alpha, beta, omega_aero, Va_norm, delta)
        self.CF = CF
        self.CM = CM

        # ---------- Compute forces and moments in the "aero" frame ---------- #
        F_aero = 0.5 * rho * geom.S * CF[0] * Va_norm ** 2
        M_aero = 0.5 * rho * geom.S * (CM[0] * [geom.b, geom.c, geom.b]) * Va_norm ** 2          # F, CF, alpha, beta, Va = compute_aero_forces(x, geom, wind_model)

        # ---------- Compute forces and moments to "earth" frame ---------- #
        F_earth = np.matmul(R_aero, F_aero)
        M_earth = np.matmul(R_aero, M_aero)

        # ---------- Compute forces and moments to "body" frame ---------- #
        F_body  = np.matmul(R_body.T, F_earth)
        M_body  = np.matmul(R_body.T, M_earth)

        # ---------- Output forces and moments ---------- #
        self.F = {"aero":F_aero, "earth":F_earth, "body":F_body}
        self.M = {"aero":M_aero, "earth":M_earth, "body":M_body}

# -------------------------- MegAWES aircraft instance -------------------------- #
class MegAWES:
    '''

    '''
    def __init__(self, t, x, wind_model = 'uniform'):
        self.t = t
        self.x = x
        self.geom = Geometry()
        self.aero = Aerodynamics(x, self.geom, wind_model)
