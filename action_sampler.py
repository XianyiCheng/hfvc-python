import numpy as np
import cvxpy as cp
from scipy.spatial.transform import Rotation
import scipy as sp

def nullspace(A, atol=1e-13, rtol=0):
    A = np.atleast_2d(A)
    u, s, vh = np.linalg.svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    # If A is an array with shape (m, k), 
    # then ns will be an array with shape (k, n), 
    # where n is the estimated dimension of the nullspace of A
    return ns

def Rot2Quat(R_o):
    # return quaternion in [qw, qx, qy, qz]
    r = Rotation.from_matrix(R_o)
    q = r.as_quat()
    return np.array([q[3],q[0],q[1],q[2]])


# Quaternion Utilities
def hat(w):
    W = np.array(
        [[0, -w[2], w[1]],
        [w[2], 0, -w[0]],
        [-w[1], w[0], 0]])
    return W

def z_normal_to_R(z):
    # return R, R*[0,0,1]' = z
    rotvec = np.cross(np.array([0,0,1]),z)
    a = np.linalg.norm(rotvec)
    
    if (a < 1e-6): 
        if (z[2] > 0):
            return np.identity(3)
        else:
            return np.array([[-1,0,0],[0,1,0],[0,0,-1]])
    
    rotvec = rotvec/a
    rotangle = np.arccos(np.dot(z, np.array([0,0,1]))/np.linalg.norm(z))
    
    W = hat(rotvec)

    R = np.identity(3) + W*np.sin(rotangle) + np.dot(W, W)*(1-np.cos(rotangle))

    return R
        

# q: qw, qx, qy, qz
def L(q):
    Lq = np.zeros((4,4))
    Lq[0,0] = q[0]
    Lq[0,1:4] = -q[1:4]
    Lq[1:4, 0] = q[1:4]
    Lq[1:4, 1:4] = q[0]*np.identity(3) + hat(q[1:4])
    return Lq

def R(q):
    Rq = np.zeros((4,4))
    Rq[0,0] = q[0]
    Rq[0,1:4] = -q[1:4]
    Rq[1:4, 0] = q[1:4]
    Rq[1:4, 1:4] = q[0]*np.identity(3) - hat(q[1:4])
    return Rq

H = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]])

def quaternion_conjugate(q):
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
    return q_conj

def Rot(q):
    Rq = np.linalg.multi_dot([np.transpose(H),L(q),np.transpose(R(q)),H])
    return Rq

def SE3Inv(T):
    T_inv = np.identity(4)
    T_inv[0:3,0:3] = np.transpose(T[0:3,0:3])
    T_inv[0:3, 3] = -np.dot(T_inv[0:3,0:3], T[0:3,3])
    return T_inv

def matrix_diag(matrices):
    n_rows = 0
    n_cols = 0
    for m in matrices:
        if len(m.shape) == 1:
            n_rows = n_rows + 1
            n_cols = n_cols + m.shape[0]
        else:   
            n_rows = n_rows + m.shape[0]
            n_cols = n_cols + m.shape[1]
    
    M = np.zeros((n_rows, n_cols))
    
    i_rows = 0
    i_cols = 0
    for m in matrices:
        if len(m.shape) == 1:
            M[i_rows:i_rows+1, i_cols:i_cols+m.shape[0]] = m
            i_rows = i_rows + 1
            i_cols = i_cols + m.shape[0]
        else:
            M[i_rows:i_rows+m.shape[0], i_cols:i_cols+m.shape[1]] = m
            i_rows = i_rows + m.shape[0]
            i_cols = i_cols + m.shape[1]
    return M


# problem specific

# configuration "x" 
# x = p_WO (object position), q_WO(object_orientation), p_WH (hand position), q_WH (hand orientation)

# generalize velocity
# v = v_o (object body twist), v_h(hand body twist)

# calculate the velocity jacobian of v and q_dot
# x_dot = Omega(x)*v
def Omega(x):
    p_WO = x[0:3]
    q_WO = x[3:7]
    p_WH = x[7:10]
    q_WH = x[10:14]

    R_WH = Rot(q_WH)
    R_WO = Rot(q_WO)
    O = matrix_diag((R_WO, 0.5*np.dot(L(q_WO),H), R_WH, 0.5*np.dot(L(q_WH),H)))
    return O

# calculate the goal specification
# Gv = bG
def goal_velocity(xo_start, xo_goal, steps):
    q_s = xo_start[3:7]
    q_g = xo_goal[3:7]
    dq = np.dot(L(q_g), quaternion_conjugate(q_s))
    angle = np.arccos(dq[0])*2
    axis = (1/np.sin(angle/2))*dq[1:4] 
    w_o = (angle/steps)*axis
    v_o = Rot(q_s).T*(xo_goal[0:3] - xo_start[0:3])/steps
    G = np.hstack([np.identity(6), np.zeros(6,6)])
    bG = np.vstack([v_o, w_o])
    return G, bG


'''
calculate the constraint jacobians of the contacts
# arguments
x: system state
p_h_Os: n_h x 3, hand-object contacts in the object frame
p_h_Hs: n_h x 3, hand-object contacts in the hand frame
p_e_Os: n_e x 3, env-object contacts in the object frame
p_e_Ws: n_e x 3, env-object contacts in the world frame
p_s_Os: n_s x 3, env-object sliding contacts in the object frame
p_s_Ws: n_s x 3, env-object sliding contacts in the world frame
n_s_Ws: n_s x 3, env-object sliding contact normals in the world frame
t_s_Ws: n_s x 3, env-object sliding contact tangent directions in the world frame
'''
def natural_constraints_jacobian(x, p_h_Os, p_h_Hs, p_e_Os, p_s_Os, n_s_Ws):
    n_h = p_h_Os.shape[0]
    n_e = p_e_Os.shape[0]
    n_s = p_s_Os.shape[0]

    Js = np.zeros((n_h*3+n_e*3+n_s, 14))

    for k in range(n_h):
        p_O = p_h_Os[k]
        p_H = p_h_Hs[k]
        J = np.array(
            [[1, 0, 0, 2*p_O[0]*x[3] - 2*p_O[1]*x[6] + 2*p_O[2]*x[5], 2*p_O[0]*x[4] + 2*p_O[1]*x[5] + 2*p_O[2]*x[6], 2*p_O[1]*x[4] - 2*p_O[0]*x[5] + 2*p_O[2]*x[3], 2*p_O[2]*x[4] - 2*p_O[0]*x[6] - 2*p_O[1]*x[3], -1,  0,  0, 2*p_H[1]*x[13] - 2*p_H[0]*x[10] - 2*p_H[2]*x[12], - 2*p_H[0]*x[11] - 2*p_H[1]*x[12] - 2*p_H[2]*x[13],   2*p_H[0]*x[12] - 2*p_H[1]*x[11] - 2*p_H[2]*x[10],   2*p_H[1]*x[10] + 2*p_H[0]*x[13] - 2*p_H[2]*x[11]],
            [0, 1, 0, 2*p_O[1]*x[3] + 2*p_O[0]*x[6] - 2*p_O[2]*x[4], 2*p_O[0]*x[5] - 2*p_O[1]*x[4] - 2*p_O[2]*x[3], 2*p_O[0]*x[4] + 2*p_O[1]*x[5] + 2*p_O[2]*x[6], 2*p_O[0]*x[3] - 2*p_O[1]*x[6] + 2*p_O[2]*x[5],  0, -1,  0, 2*p_H[2]*x[11] - 2*p_H[0]*x[13] - 2*p_H[1]*x[10],   2*p_H[1]*x[11] - 2*p_H[0]*x[12] + 2*p_H[2]*x[10], - 2*p_H[0]*x[11] - 2*p_H[1]*x[12] - 2*p_H[2]*x[13],   2*p_H[1]*x[13] - 2*p_H[0]*x[10] - 2*p_H[2]*x[12]],
            [0, 0, 1, 2*p_O[1]*x[4] - 2*p_O[0]*x[5] + 2*p_O[2]*x[3], 2*p_O[1]*x[3] + 2*p_O[0]*x[6] - 2*p_O[2]*x[4], 2*p_O[1]*x[6] - 2*p_O[0]*x[3] - 2*p_O[2]*x[5], 2*p_O[0]*x[4] + 2*p_O[1]*x[5] + 2*p_O[2]*x[6],  0,  0, -1, 2*p_H[0]*x[12] - 2*p_H[1]*x[11] - 2*p_H[2]*x[10],   2*p_H[2]*x[11] - 2*p_H[0]*x[13] - 2*p_H[1]*x[10],   2*p_H[0]*x[10] - 2*p_H[1]*x[13] + 2*p_H[2]*x[12], - 2*p_H[0]*x[11] - 2*p_H[1]*x[12] - 2*p_H[2]*x[13]]])
 
        Js[3*k:3*k+3,:] = J
    
    for k in range(n_e):
        p_O = p_e_Os[k]
        J = np.array([
            [1, 0, 0, 2*p_O[0]*x[3] - 2*p_O[1]*x[6] + 2*p_O[2]*x[5], 2*p_O[0]*x[4] + 2*p_O[1]*x[5] + 2*p_O[2]*x[6], 2*p_O[1]*x[4] - 2*p_O[0]*x[5] + 2*p_O[2]*x[3], 2*p_O[2]*x[4] - 2*p_O[0]*x[6] - 2*p_O[1]*x[3], 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 2*p_O[1]*x[3] + 2*p_O[0]*x[6] - 2*p_O[2]*x[4], 2*p_O[0]*x[5] - 2*p_O[1]*x[4] - 2*p_O[2]*x[3], 2*p_O[0]*x[4] + 2*p_O[1]*x[5] + 2*p_O[2]*x[6], 2*p_O[0]*x[3] - 2*p_O[1]*x[6] + 2*p_O[2]*x[5], 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 2*p_O[1]*x[4] - 2*p_O[0]*x[5] + 2*p_O[2]*x[3], 2*p_O[1]*x[3] + 2*p_O[0]*x[6] - 2*p_O[2]*x[4], 2*p_O[1]*x[6] - 2*p_O[0]*x[3] - 2*p_O[2]*x[5], 2*p_O[0]*x[4] + 2*p_O[1]*x[5] + 2*p_O[2]*x[6], 0, 0, 0, 0, 0, 0, 0]])
        Js[n_h*3+3*k:n_h*3+3*k+3,:] = J
    
    for k in range(n_s):
        p_O = p_s_Os[k]
        n_W = n_s_Ws[k]
        J = np.array([n_W[0], n_W[1], n_W[2], n_W[0]*(2*p_O[0]*x[3] - 2*p_O[1]*x[6] + 2*p_O[2]*x[5]) + n_W[1]*(2*p_O[1]*x[3] + 2*p_O[0]*x[6] - 2*p_O[2]*x[4]) + n_W[2]*(2*p_O[1]*x[4] - 2*p_O[0]*x[5] + 2*p_O[2]*x[3]), n_W[0]*(2*p_O[0]*x[4] + 2*p_O[1]*x[5] + 2*p_O[2]*x[6]) - n_W[1]*(2*p_O[1]*x[4] - 2*p_O[0]*x[5] + 2*p_O[2]*x[3]) + n_W[2]*(2*p_O[1]*x[3] + 2*p_O[0]*x[6] - 2*p_O[2]*x[4]), n_W[0]*(2*p_O[1]*x[4] - 2*p_O[0]*x[5] + 2*p_O[2]*x[3]) + n_W[1]*(2*p_O[0]*x[4] + 2*p_O[1]*x[5] + 2*p_O[2]*x[6]) - n_W[2]*(2*p_O[0]*x[3] - 2*p_O[1]*x[6] + 2*p_O[2]*x[5]), n_W[1]*(2*p_O[0]*x[3] - 2*p_O[1]*x[6] + 2*p_O[2]*x[5]) - n_W[0]*(2*p_O[1]*x[3] + 2*p_O[0]*x[6] - 2*p_O[2]*x[4]) + n_W[2]*(2*p_O[0]*x[4] + 2*p_O[1]*x[5] + 2*p_O[2]*x[6]), 0, 0, 0, 0, 0, 0, 0])
        Js[(n_h+n_e)*3+k,:] = J
    
    return Js

'''
calculate the constraint jacobians of the contacts
# arguments
x: system state
p_h_Os: n_h x 3, hand-object contacts in the object frame
p_h_Hs: n_h x 3, hand-object contacts in the hand frame
n_h_Ws: n_h x 3, hand-object contact normals in the world frame
p_e_Os: n_e x 3, env-object contacts in the object frame
p_e_Ws: n_e x 3, env-object contacts in the world frame
n_e_Ws: n_e x 3, env-object contact normals in the world frame
p_s_Os: n_s x 3, env-object sliding contacts in the object frame
p_s_Ws: n_s x 3, env-object sliding contacts in the world frame
n_s_Ws: n_s x 3, env-object sliding contact normals in the world frame
t_s_Ws: n_s x 3, env-object sliding contact tangent directions in the world frame
mu_h: friction coefficient of the hand-object contacts
mu_e: friction coefficent of the env-object contacts
'''

def force_constraints(x, p_h_Os, p_h_Hs, n_h_Ws, p_e_Os, n_e_Ws, p_s_Os, n_s_Ws, t_s_Ws, mu_h, mu_e):
    
    n_h = p_h_Os.shape[0]
    n_e = p_e_Os.shape[0]
    n_s = p_s_Os.shape[0]

    # size of reaction forces lambda: 
    n = n_h*3 + n_e*3 + n_s*2

    FC_h = np.array([[0,0,-1],[-1,-1,-mu_h],[1,-1,-mu_h],[-1,1,-mu_h], [1,1,-mu_h]])
    FC_e = np.array([[0,0,-1],[-1,-1,-mu_e],[1,-1,-mu_e],[-1,1,-mu_e], [1,1,-mu_e]])
    FC_s = np.array([[-1,0],[0,1],[-mu_e,-1]])

    J_Ts = ()

    FCs = ()


    for k in range(n_h):
        p_O = p_h_Os[k]
        p_H = p_h_Hs[k]
        J = np.array(
            [[1, 0, 0, 2*p_O[0]*x[3] - 2*p_O[1]*x[6] + 2*p_O[2]*x[5], 2*p_O[0]*x[4] + 2*p_O[1]*x[5] + 2*p_O[2]*x[6], 2*p_O[1]*x[4] - 2*p_O[0]*x[5] + 2*p_O[2]*x[3], 2*p_O[2]*x[4] - 2*p_O[0]*x[6] - 2*p_O[1]*x[3], -1,  0,  0, 2*p_H[1]*x[13] - 2*p_H[0]*x[10] - 2*p_H[2]*x[12], - 2*p_H[0]*x[11] - 2*p_H[1]*x[12] - 2*p_H[2]*x[13],   2*p_H[0]*x[12] - 2*p_H[1]*x[11] - 2*p_H[2]*x[10],   2*p_H[1]*x[10] + 2*p_H[0]*x[13] - 2*p_H[2]*x[11]],
            [0, 1, 0, 2*p_O[1]*x[3] + 2*p_O[0]*x[6] - 2*p_O[2]*x[4], 2*p_O[0]*x[5] - 2*p_O[1]*x[4] - 2*p_O[2]*x[3], 2*p_O[0]*x[4] + 2*p_O[1]*x[5] + 2*p_O[2]*x[6], 2*p_O[0]*x[3] - 2*p_O[1]*x[6] + 2*p_O[2]*x[5],  0, -1,  0, 2*p_H[2]*x[11] - 2*p_H[0]*x[13] - 2*p_H[1]*x[10],   2*p_H[1]*x[11] - 2*p_H[0]*x[12] + 2*p_H[2]*x[10], - 2*p_H[0]*x[11] - 2*p_H[1]*x[12] - 2*p_H[2]*x[13],   2*p_H[1]*x[13] - 2*p_H[0]*x[10] - 2*p_H[2]*x[12]],
            [0, 0, 1, 2*p_O[1]*x[4] - 2*p_O[0]*x[5] + 2*p_O[2]*x[3], 2*p_O[1]*x[3] + 2*p_O[0]*x[6] - 2*p_O[2]*x[4], 2*p_O[1]*x[6] - 2*p_O[0]*x[3] - 2*p_O[2]*x[5], 2*p_O[0]*x[4] + 2*p_O[1]*x[5] + 2*p_O[2]*x[6],  0,  0, -1, 2*p_H[0]*x[12] - 2*p_H[1]*x[11] - 2*p_H[2]*x[10],   2*p_H[2]*x[11] - 2*p_H[0]*x[13] - 2*p_H[1]*x[10],   2*p_H[0]*x[10] - 2*p_H[1]*x[13] + 2*p_H[2]*x[12], - 2*p_H[0]*x[11] - 2*p_H[1]*x[12] - 2*p_H[2]*x[13]]])
        
        n_W = n_h_Ws[k]
        R_WC = z_normal_to_R(n_W)
        J_T = np.dot(J.T, R_WC) # 14x3

        J_Ts = J_Ts + (J_T,)

        FCs = FCs + (FC_h,)

    for k in range(n_e):
        p_O = p_e_Os[k]
        J = np.array([
            [1, 0, 0, 2*p_O[0]*x[3] - 2*p_O[1]*x[6] + 2*p_O[2]*x[5], 2*p_O[0]*x[4] + 2*p_O[1]*x[5] + 2*p_O[2]*x[6], 2*p_O[1]*x[4] - 2*p_O[0]*x[5] + 2*p_O[2]*x[3], 2*p_O[2]*x[4] - 2*p_O[0]*x[6] - 2*p_O[1]*x[3], 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 2*p_O[1]*x[3] + 2*p_O[0]*x[6] - 2*p_O[2]*x[4], 2*p_O[0]*x[5] - 2*p_O[1]*x[4] - 2*p_O[2]*x[3], 2*p_O[0]*x[4] + 2*p_O[1]*x[5] + 2*p_O[2]*x[6], 2*p_O[0]*x[3] - 2*p_O[1]*x[6] + 2*p_O[2]*x[5], 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 2*p_O[1]*x[4] - 2*p_O[0]*x[5] + 2*p_O[2]*x[3], 2*p_O[1]*x[3] + 2*p_O[0]*x[6] - 2*p_O[2]*x[4], 2*p_O[1]*x[6] - 2*p_O[0]*x[3] - 2*p_O[2]*x[5], 2*p_O[0]*x[4] + 2*p_O[1]*x[5] + 2*p_O[2]*x[6], 0, 0, 0, 0, 0, 0, 0]])
        
        n_W = n_e_Ws[k]
        R_WC = z_normal_to_R(n_W)
        J_T = np.dot(J.T, R_WC) # 14x3

        J_Ts = J_Ts + (J_T,)

        FCs = FCs + (FC_e,)
    
    for k in range(n_s):
        p_O = p_s_Os[k]
        n_W = n_s_Ws[k]
        t_W = t_s_Ws[k]
        J = np.array([
            [1, 0, 0, 2*p_O[0]*x[3] - 2*p_O[1]*x[6] + 2*p_O[2]*x[5], 2*p_O[0]*x[4] + 2*p_O[1]*x[5] + 2*p_O[2]*x[6], 2*p_O[1]*x[4] - 2*p_O[0]*x[5] + 2*p_O[2]*x[3], 2*p_O[2]*x[4] - 2*p_O[0]*x[6] - 2*p_O[1]*x[3], 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 2*p_O[1]*x[3] + 2*p_O[0]*x[6] - 2*p_O[2]*x[4], 2*p_O[0]*x[5] - 2*p_O[1]*x[4] - 2*p_O[2]*x[3], 2*p_O[0]*x[4] + 2*p_O[1]*x[5] + 2*p_O[2]*x[6], 2*p_O[0]*x[3] - 2*p_O[1]*x[6] + 2*p_O[2]*x[5], 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 2*p_O[1]*x[4] - 2*p_O[0]*x[5] + 2*p_O[2]*x[3], 2*p_O[1]*x[3] + 2*p_O[0]*x[6] - 2*p_O[2]*x[4], 2*p_O[1]*x[6] - 2*p_O[0]*x[3] - 2*p_O[2]*x[5], 2*p_O[0]*x[4] + 2*p_O[1]*x[5] + 2*p_O[2]*x[6], 0, 0, 0, 0, 0, 0, 0]])
        
        R_WC = z_normal_to_R(n_W)

        J_T_n = np.reshape(np.dot(np.dot(J.T, R_WC), np.array([0,0,1])),(-1,1))

        T_WC = z_normal_to_R(t_W)

        J_T_t = np.reshape(np.dot(np.dot(J.T, T_WC), np.array([0,0,1])),(-1,1))

        J_Ts = J_Ts + (J_T_n, J_T_t)

        FCs = FCs + (FC_s,)
    

    NT = np.hstack(J_Ts)
    A_lambda = matrix_diag(FCs)
    b_lambda = np.zeros(n)

    return NT, A_lambda

'''
hfvc: compute the hybrid force velocity control
# Arguments:
* 

Newton: T*N_force_T*\lambda + \ita + T*F = 0
Force constraints: Aeq * [\lambda; \ita] = beq, A*[\lambda; \ita] <= b_A

# Output:

'''

def hfvc(N_all, G, b_G, N_force_T, F, Aeq, beq, A, b_A):
    kDimActualized = 6
    kDimUnActualized = 6
    
    # kDimLambda: total dimension of constraint force (not including sliding forces)
    kDimLambda = N_all.shape[0] 
    kDimGeneralized  = kDimActualized + kDimUnActualized

    NG = np.vstack((N_all, G))

    rank_N = np.linalg.matrix_rank(N_all)

    rank_NG = np.linalg.matrix_rank(NG)

    n_av_min = rank_NG - rank_N # forcefull manipulation, do force control in all the directions that goal velocity is not specified & in the Jacobian space of natural constriants 

    n_av_max = kDimGeneralized - rank_N # max velocity control, do force control only in the Jacobian space of natural constriants 

    n_av = n_av_min # dim of velocity control, 
    n_af = kDimActualized - n_av # dim of force control

    if n_av_min == 0:
        print("the goal conflict with the natural constraints, problem infeasible\n")
        return False
    
    # --- Velocity control computation 

    # free generalized motion space U
    U = np.transpose(nullspace(N_all))
    Sa = matrix_diag((np.zeros((6,6)), np.identity(6)))

    # U_bar is the rowspace of U*Sa
    U_bar = sp.linalg.orth(np.dot(U, Sa).T).T

    if U_bar.shape[0] < n_av:
        print("not enough free motion space to achieve the goal, the problem is infeasible\n")
        return False
    else:
        
        K = np.transpose(nullspace(np.transpose(np.dot(U_bar, nullspace(NG)))))
        
        if K.shape[0] >= n_av:
            K = K[0:n_av,:]
            C = np.dot(K, U_bar)
        else:
            # TODO: figure out why? what data?
            # minimum velocity control cannot satisfy the velocity constraints and goal 
            print("Switch to maximum velocity control. Not enough free motion space do minimum velocity control(cannot do force control in directions that have no goal velocity specification) \n")
            n_av = U_bar.shape[0]
            n_af = kDimActualized - n_av
            C = U_bar
        
    # the actuated part of velocity control in the generalized velocity space (actuated part)
    # each row of Rc is a velocity control axis in the hand body velocity space
    Rc = C[:, 6:]
     
    Ra = np.vstack((np.transpose(nullspace(Rc)),Rc))

    b_NG = np.hstack((np.zeros(N_all.shape[0]), b_G))

    v_star = np.dot(np.linalg.pinv(NG), b_NG)
    w_av = np.dot(C, v_star)

    T = matrix_diag((np.identity(6), Ra))

    # --- Velocity control computation end

    # --- Force control computation

    n_var = Aeq.shape[1]
    n_ita = kDimGeneralized
    n_lambda = n_var - n_ita

    x = cp.Variable(n_var)
    G = A
    h = b_A
    Geq = np.vstack((np.hstack((np.dot(T, N_force_T), np.identity(n_ita))), Aeq))
    heq = np.hstack((np.dot(-T, F), beq))
    
    G_ = np.vstack((G, Geq, -Geq))
    h_ = np.hstack((h, heq+1e-6, -heq+1e-6))
    
    P = np.identity(n_var) # TODO: maintain certain amount of contact forces
    q = np.zeros(n_var)

    prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P) + q.T @ x), [G @x <= h, Geq @x == heq])
    # prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P) + q.T @ x), [G_ @x <= h_])
    prob.solve()

    if prob.status not in ["infeasible", "unbounded"]:
        x_sol = x.value
        ita_f = x_sol[n_lambda+6:n_lambda+6+n_af]
        return n_av, n_af, w_av, ita_f, Ra
    else:
        print("no force solution, problem infeasible\n")
        return False


def compute_hfvc(T_WO, T_WO_goal, T_WH_goal, object_env_cts, object_env_cts_goal, robot_obj_cts, mu_h, mu_e, F_external):
    
    # make sure points are in the right format
    object_env_cts_local = np.reshape(object_env_cts['ct_pts_local'],(-1,3))
    object_env_cts_world = np.reshape(object_env_cts['ct_pts_world'],(-1,3))
    object_env_cts_local_normals = np.reshape(object_env_cts['ct_normals_local'],(-1,3))
    object_env_cts_world_normals = np.reshape(object_env_cts['ct_normals_world'],(-1,3))

    object_env_cts_local_goal = np.reshape(object_env_cts_goal['ct_pts_local'],(-1,3))
    object_env_cts_world_goal = np.reshape(object_env_cts_goal['ct_pts_world'],(-1,3))
    object_env_cts_local_normals_goal = np.reshape(object_env_cts_goal['ct_normals_local'],(-1,3))
    object_env_cts_world_normals_goal = np.reshape(object_env_cts_goal['ct_normals_world'],(-1,3))

    robot_obj_cts_world_final = np.reshape(robot_obj_cts['final_ct_pts_world'], (-1,3))

    # compute G, bG from T_WO and T_WO_goal
    T_b = np.dot(SE3Inv(T_WO),T_WO_goal)
    R_b = Rotation.from_matrix(T_b[0:3,0:3])
    w_b = R_b.as_rotvec()
    v_b = T_b[0:3,3]
    bG = np.hstack((v_b, w_b))
    G = np.hstack((np.identity(6), np.zeros((6,6)))) # TODO: is that possible to have less G? not specify the full object velocity

    # compute x, p_h_Os, p_h_Hs, n_h_Ws, p_e_Os, n_e_Ws, p_s_Os, n_s_Ws, t_s_Ws
    T_OH = np.dot(SE3Inv(T_WO_goal), T_WH_goal)
    T_WH = np.dot(T_WO, T_OH)
    x = np.zeros(14)
    x[0:3] = T_WO[0:3,3]
    x[3:7] = Rot2Quat(T_WO[0:3,0:3])
    x[7:10] = T_WH[0:3,3]
    x[10:] = Rot2Quat(T_WH[0:3,0:3])

    # TODO: is point contact model enough for hand-object contacts?
    
    p_h_Os = np.reshape(robot_obj_cts['ct_normal_local'], (-1, 3))
    n_h_Ws = np.reshape(robot_obj_cts['init_ct_normal_world'], (-1, 3))
    p_h_Ws_homo = np.hstack((robot_obj_cts_world_final, np.ones((p_h_Os.shape[0],1))))
    p_h_Hs = np.dot(SE3Inv(T_WH_goal), p_h_Ws_homo.T).T
    p_h_Hs = p_h_Hs[:,0:3]

    # TODO: improve decide which object-env is sticking/sliding, may need some clustering method
    mask = object_env_cts['cts_mask'] & object_env_cts_goal['cts_mask']
    dps = object_env_cts_world_goal - object_env_cts_world
    ds = np.linalg.norm(dps, axis=1)
    mask_stick = (ds <= 1e-5) & mask
    mask_slide = (ds > 1e-5) & mask

    p_e_Os = object_env_cts_local[mask_stick]
    n_e_Ws = object_env_cts_world[mask_stick] 
    p_s_Os = object_env_cts_local[mask_slide]
    n_s_Ws = object_env_cts_world_normals[mask_slide]
    t_s_Ws = object_env_cts_world_goal[mask_slide] - object_env_cts_world[mask_slide]
    t_s_Ws = np.array([t/np.linalg.norm(t) for t in t_s_Ws]) #TODO: do we need a cone for sliding directions? not so exact?

    # compute N_all, N_force_T
    O = Omega(x)
    J_all = natural_constraints_jacobian(x, p_h_Os, p_h_Hs, p_e_Os, p_s_Os, n_s_Ws)
    J_force_T, A_lambda = force_constraints(x, p_h_Os, p_h_Hs, n_h_Ws, p_e_Os, n_e_Ws, p_s_Os, n_s_Ws, t_s_Ws, mu_h, mu_e)
    N_all = np.dot(J_all, O)
    N_force_T = np.dot(O.T, J_force_T)

    n_lambda = A_lambda.shape[1]
    n_ita = 12

    # no unactuated force: A_u * ita = 0
    Aeq = np.hstack((np.zeros((6, n_lambda)),np.identity(6), np.zeros((6,6))))
    beq = np.zeros(6)
    
    # compute Aeq, beq, A, b_A
    A = np.hstack((A_lambda, np.zeros((A_lambda.shape[0], n_ita))))
    b_A = np.zeros(A_lambda.shape[0])

    results = hfvc(N_all, G, bG, N_force_T, F_external, Aeq, beq, A, b_A)
    return results

    
def hybrid_force_velocity_control(params):

    (init_obj_pose, final_obj_pose, final_ee_pose, 
    obj_env_ct_pts_local, obj_env_ct_pts_world, obj_env_ct_normal_world, obj_env_ct_mask, 
    obj_env_cts_pts_world_goal, obj_env_ct_mask_goal, 
    robot_obj_ct_normals_object, robot_obj_ct_normals_world_init, robot_obj_ct_pts_world_final) = params

    # TODO: these should from params
    mu_h = 0.8
    mu_e = 0.5
    F_external = np.array([0,0,-9.8,0,0,0, 0,0,0,0,0,0]) 

    T_WO = np.identity(4)
    T_WO[0:3,0:3] = init_obj_pose.rotation
    T_WO[0:3, 3] = init_obj_pose.translation

    T_WO_goal = np.identity(4)
    T_WO_goal[0:3,0:3] = final_obj_pose.rotation
    T_WO_goal[0:3, 3] = final_obj_pose.translation

    T_WH_goal = np.identity(4)
    T_WH_goal[0:3,0:3] = final_ee_pose.rotation
    T_WH_goal[0:3, 3] = final_ee_pose.translation

        # make sure points are in the right format
    object_env_cts_local = np.reshape(obj_env_ct_pts_local,(-1,3))
    object_env_cts_world = np.reshape(obj_env_ct_pts_world,(-1,3))
    object_env_cts_world_normals = np.reshape(obj_env_ct_normal_world,(-1,3))

    object_env_cts_world_goal = np.reshape(obj_env_cts_pts_world_goal,(-1,3))

    robot_obj_cts_world_final = np.reshape(robot_obj_ct_pts_world_final, (-1,3))

    # compute G, bG from T_WO and T_WO_goal
    T_b = np.dot(SE3Inv(T_WO),T_WO_goal)
    R_b = Rotation.from_matrix(T_b[0:3,0:3])
    w_b = R_b.as_rotvec()
    v_b = T_b[0:3,3]
    bG = np.hstack((v_b, w_b))
    G = np.hstack((np.identity(6), np.zeros((6,6)))) # TODO: is that possible to have less G? not specify the full object velocity

    # compute x, p_h_Os, p_h_Hs, n_h_Ws, p_e_Os, n_e_Ws, p_s_Os, n_s_Ws, t_s_Ws
    T_OH = np.dot(SE3Inv(T_WO_goal), T_WH_goal)
    T_WH = np.dot(T_WO, T_OH)
    x = np.zeros(14)
    x[0:3] = T_WO[0:3,3]
    x[3:7] = Rot2Quat(T_WO[0:3,0:3])
    x[7:10] = T_WH[0:3,3]
    x[10:] = Rot2Quat(T_WH[0:3,0:3])

    # TODO: is point contact model enough for hand-object contacts?
    
    p_h_Os = np.reshape(robot_obj_ct_normals_object, (-1, 3))
    n_h_Ws = np.reshape(robot_obj_ct_normals_world_init, (-1, 3))
    p_h_Ws_homo = np.hstack((robot_obj_cts_world_final, np.ones((p_h_Os.shape[0],1))))
    p_h_Hs = np.dot(SE3Inv(T_WH_goal), p_h_Ws_homo.T).T
    p_h_Hs = p_h_Hs[:,0:3]

    # TODO: improve decide which object-env is sticking/sliding, may need some clustering method
    mask = obj_env_ct_mask & obj_env_ct_mask_goal
    dps = object_env_cts_world_goal - object_env_cts_world
    ds = np.linalg.norm(dps, axis=1)
    mask_stick = (ds <= 1e-5) & mask
    mask_slide = (ds > 1e-5) & mask

    p_e_Os = object_env_cts_local[mask_stick]
    n_e_Ws = object_env_cts_world[mask_stick] 
    p_s_Os = object_env_cts_local[mask_slide]
    n_s_Ws = object_env_cts_world_normals[mask_slide]
    t_s_Ws = object_env_cts_world_goal[mask_slide] - object_env_cts_world[mask_slide]
    t_s_Ws = np.array([t/np.linalg.norm(t) for t in t_s_Ws]) #TODO: do we need a cone for sliding directions? not so exact?

    # compute N_all, N_force_T
    O = Omega(x)
    J_all = natural_constraints_jacobian(x, p_h_Os, p_h_Hs, p_e_Os, p_s_Os, n_s_Ws)
    J_force_T, A_lambda = force_constraints(x, p_h_Os, p_h_Hs, n_h_Ws, p_e_Os, n_e_Ws, p_s_Os, n_s_Ws, t_s_Ws, mu_h, mu_e)
    N_all = np.dot(J_all, O)
    N_force_T = np.dot(O.T, J_force_T)

    n_lambda = A_lambda.shape[1]
    n_ita = 12

    # no unactuated force: A_u * ita = 0
    Aeq = np.hstack((np.zeros((6, n_lambda)),np.identity(6), np.zeros((6,6))))
    beq = np.zeros(6)
    
    # compute Aeq, beq, A, b_A
    A = np.hstack((A_lambda, np.zeros((A_lambda.shape[0], n_ita))))
    b_A = np.zeros(A_lambda.shape[0])

    results = hfvc(N_all, G, bG, N_force_T, F_external, Aeq, beq, A, b_A)

    return results

            
    

