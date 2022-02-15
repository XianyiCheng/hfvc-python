function Rq = R(qw, qx, qy, qz)

Rq = [qw -qx -qy -qz; [qx;qy;qz] qw*eye(3) - hat([qx;qy;qz])];

end

