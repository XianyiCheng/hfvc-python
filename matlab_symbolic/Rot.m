function Rotq = Rot(qw, qx, qy, qz)

H = [0 0 0; eye(3)];
Rotq = H'*L(qw, qx, qy, qz)*R(qw, qx, qy, qz)'*H;

end

