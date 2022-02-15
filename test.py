from action_sampler import *

np.set_printoptions(precision=4,suppress=True, linewidth=120)
q = np.array([1,2,3,4])
q_conj = np.array([q[0], -q[1], -q[2], -q[3]])

# test hat
print("hat", hat(q[1:4]))
# test L
print("L", L(q))
# test R
print("R", R(q))
# test H
print("H", H)
print(np.linalg.multi_dot([np.transpose(H),L(q),np.transpose(R(q)),H]))
print(np.linalg.multi_dot([np.transpose(H),L(q_conj),np.transpose(R(q_conj)),H]).T)

# test matrix_diag
print("matrix_diag\n", matrix_diag((np.identity(2), 0.2+np.zeros((3,2)), 5*np.identity(3))))

print("test nullspace \n")
print(nullspace(np.array([1,0,0,0])))
A = np.array([[1,0,0,0],[2,2,4,0]])
print(nullspace(A),"\n", np.dot(A, nullspace(A)))

print("test Rot2Quat\n")
R_t = np.array([[ 0.      ,  0.733844, -0.679318],
       [-0.      ,  0.679318,  0.733844],
       [ 1.      ,  0.      ,  0.      ]])
print(Rot2Quat(R_t))
print(Rot2Quat(np.identity(3)))

print("test z_normal_to_R \n")
print(z_normal_to_R(np.array([1,0,0])))
print(z_normal_to_R(np.array([0,1,0])))
print(z_normal_to_R(np.array([0,0,1])))
print(z_normal_to_R(np.array([0,0,-1])))
print(z_normal_to_R(np.array([1,-1,1])))
Rz = z_normal_to_R(np.array([1,-1,1]))
print(np.dot(Rz.T,Rz))

# test SE3Inv
T = np.identity(4)
T[0:3,0:3] = np.array([[ 0.      ,  0.733844, -0.679318],
       [-0.      ,  0.679318,  0.733844],
       [ 1.      ,  0.      ,  0.      ]])
T[0:3,3] = np.array([0.374174, 0.081769, 0.307227])
print("SE3Inv :", np.sum(np.dot(T, SE3Inv(T)) - np.identity(4)))

print("test Omega")
x = np.array([0,0,0, -0.64794252 , 0.28314395,  0.64794252,  0.28314395, 1,0,1,1,0,0,0])
print(Omega(x).shape)
print(Omega(x))

# test goal_velocity 

print("test natural_constraints_jacobian\n")
x = np.array([0,0,1,1,0,0,0,-1,0,2,1,0,0,0])
p_s_Os = np.array([[1,-1,-1],[1,1,-1],[-1,-1,-1],[-1,1,-1]])
n_s_Ws = np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1]])
p_e_Os = np.zeros((0,3))
p_h_Os = np.array([[-1,0,0],[0,0,1]])
p_h_Hs = np.array([[0,0,-1],[1,0,0]])
n_h_Ws = np.array([[1,0,0],[0,0,-1]])
n_e_Ws = np.zeros((0,3))
t_s_Ws = np.array([[1,0,0],[1,0,0],[1,0,0],[1,0,0]])
mu_h = 0.8
mu_e = 0.3
J_all = natural_constraints_jacobian(x, p_h_Os, p_h_Hs, p_e_Os, p_s_Os, n_s_Ws)
v1 = np.array([1,0,0,0,0,0,1,0,0,0,0,0])
print(np.dot(np.dot(J_all, Omega(x)), v1))


print("test force_constraints")
J_force_T, A_lambda = force_constraints(x, p_h_Os, p_h_Hs, n_h_Ws, p_e_Os, n_e_Ws, p_s_Os, n_s_Ws, t_s_Ws, mu_h, mu_e)
print(J_force_T.shape)
print(A_lambda.shape)

#test contact force on body wrench

lambda1 = np.array([0,0,0, 0,0,0, 1,0, 1,0, 1,0, 1,0])
print(np.dot(np.dot(Omega(x).T, J_force_T), lambda1))

lambda2 = np.array([0,0,1, 0,0,0, 0,0, 0,0, 0,0, 0,0])
print(np.dot(np.dot(Omega(x).T, J_force_T), lambda2))

lambda3 = np.array([0,0,0, 0,0,1, 0,0, 0,0, 0,0, 0,0])
print(np.dot(np.dot(Omega(x).T, J_force_T), lambda3))

lambda3 = np.array([1,0,0, 0,0,0, 0,0, 0,0, 0,0, 0,0])
print(np.dot(np.dot(Omega(x).T, J_force_T), lambda3))


# hfvc
O = Omega(x)

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

G = np.hstack((np.identity(6), np.zeros((6,6))))
b_G = np.array([1,0,0,0,0,0])

F = np.array([0,0,-9.8,0,0,0, 0,0,0,0,0,0]) # external force on the generalized force coordinate, including wrenches on the object and the hand

print("test hfvc: sliding")
hfvc(N_all, G, b_G, N_force_T, F, Aeq, beq, A, b_A)

print("test hfvc: pivoting")

p_s_Os = np.zeros((0,3))
n_s_Ws = np.zeros((0,3))
t_s_Ws = np.zeros((0,3))

p_e_Os = np.array([[1,-1,-1],[1,1,-1]])
n_e_Ws = np.array([[0,0,1], [0,0,1]])

p_h_Os = np.array([[-1,0,0],[0,0,1]])
p_h_Hs = np.array([[0,0,-1],[1,0,0]])
n_h_Ws = np.array([[1,0,0],[0,0,-1]])

mu_h = 0.8
mu_e = 0.5
J_all = natural_constraints_jacobian(x, p_h_Os, p_h_Hs, p_e_Os, p_s_Os, n_s_Ws)

J_force_T, A_lambda = force_constraints(x, p_h_Os, p_h_Hs, n_h_Ws, p_e_Os, n_e_Ws, p_s_Os, n_s_Ws, t_s_Ws, mu_h, mu_e)

# hfvc
O = Omega(x)

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

G = np.array([0,0,0,0,1,0, 0,0,0,0,0,0])
b_G = np.array([0.1])

F = np.array([0,0,-9.8,0,0,0, 0,0,0,0,0,0]) # external force on the generalized force coordinate, including wrenches on the object and the hand

hfvc(N_all, G, b_G, N_force_T, F, Aeq, beq, A, b_A)
# compute_hfvc
