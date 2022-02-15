q = sym('q', [1,14], 'real');

p_WO = q(1:3);
q_WO = q(4:7);
p_WH = q(8:10);
q_WH = q(11:14);

R_WO = Rot(q_WO(1), q_WO(2), q_WO(3), q_WO(4));
R_WH = Rot(q_WH(1), q_WH(2), q_WH(3), q_WH(4));

% sticking hand-object contacts
p_h_O = sym('p_h_O', [3,1],'real'); % hand-object contact in the object frame
p_h_H = sym('p_h_H', [3,1],'real'); % hand-object contact in the hand frame

f1 = R_WO*p_h_O + p_WO' - (R_WH*p_h_H + p_WH'); % f1 = 0

% fixed env-object contacts
p_e_O = sym('p_e_O', [3,1],'real'); % env-object contact in the object frame
p_e_W = sym('p_e_W', [3,1],'real'); % env-object contact in the world frame

f2 = R_WO*p_e_O + p_WO' - p_e_W; % f2 = 0

% sliding env-object contact
n_e_W = sym('n_e_W', [3,1],'real'); % env-object contact normal in the world frame

f3 = n_e_W'*(R_WO*p_e_O + p_WO' - p_e_W);

% jacobians 
J1 = jacobian(f1, q)
J2 = jacobian(f2, q)
J3 = jacobian(f3, q)
