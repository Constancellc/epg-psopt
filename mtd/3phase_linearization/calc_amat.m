function [ Amat ] = calc_amat( Y,RRR )

n = size(Y,1);

VTV = [              eye(3),zeros(3,n-3),          zeros(3,3*n)];
VTT = [zeros(3,1*n),eye(3),zeros(3,n-3),          zeros(3,2*n)];
PQP = [zeros(n-3,2*n),zeros(n-3,3),eye(n-3), zeros(n-3,1*n)];
PQQ = [zeros(n-3,3*n),zeros(n-3,3),eye(n-3)               ];

% VTV = [kron(e0',eye(3)), zeros(3, 3*n), zeros(3, 3*n), zeros(3, 3*n)];
% VTT = [zeros(3, 3*n), kron(e0',eye(3)), zeros(3, 3*n), zeros(3, 3*n)];
% PQP = [zeros(3*(n-1),3*n), zeros(3*(n-1),3*n), zeros(3*(n-1),3), eye(3*(n-1)), zeros(3*(n-1),3*n)];
% PQQ = [zeros(3*(n-1),3*n), zeros(3*(n-1),3*n), zeros(3*(n-1),3*n), zeros(3*(n-1),3), eye(3*(n-1))];

NNN = Nmatrix(2*n);
LLL = bracket(Y);

% equivalent, when linearizing around the no load solution
Amat = [NNN*inv(RRR)*LLL*RRR eye(2*n); VTV; VTT; PQP; PQQ];

end

