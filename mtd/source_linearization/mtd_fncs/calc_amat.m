function [ A ] = calc_amat( Y,V )

n = size(Y,1);

% Linearised model
WWW = bracket(diag(conj(Y*V)));
EEE = bracket(diag(V));
NNN = Nmatrix(2*n);
LLL = bracket(Y);
RRR = Rmatrix(abs(V),angle(V));

VTV = [eye(3),           zeros(3,n-3),zeros(3,3*n)];
VTT = [zeros(3,1*n),    eye(3),         zeros(3,n-3), zeros(3,2*n)];
PQP = [zeros(n-3,2*n),  zeros(n-3,3),   eye(n-3),zeros(n-3,1*n)];
PQQ = [zeros(n-3,3*n),  zeros(n-3,3),   eye(n-3)               ];

A = [((WWW + EEE*NNN*LLL)*RRR), eye(2*n); 
        VTV; 
        VTT; 
        PQP; 
        PQQ];
end