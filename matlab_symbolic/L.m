function Lq = L(qw, qx, qy, qz)

Lq = [qw -qx -qy -qz; [qx;qy;qz] qw*eye(3) + hat([qx;qy;qz])];

end

