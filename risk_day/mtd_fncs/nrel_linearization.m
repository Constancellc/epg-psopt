function [ My,Md,a,Ky,Kd,b ] = nrel_linearization( xh,H,Y,VV )
% function [ My,Md,a,Ky,Kd,b ] = nrel_linearization( xh,H,Y,Vh,V0 )
%NREL_LINEARIZATION Summary of this function goes here
%   Detailed explanation goes here

V0 = VV(1:3);
Vh = VV(4:end);
% Nd = size(H,1);
Ny = size(H,2);
vh = abs(Vh);

xhy = xh(1:2*Ny);
xhd = xh(2*Ny+1:end);

Yll = Y(4:end,4:end);
Yl0 = Y(4:end,1:3);

w = -Yll\Yl0*V0;

% now create linearisation matrices
My0 = inv(diag(conj(Vh))*Yll);
Md0 = inv(Yll)*(H')*inv(diag(H*conj(Vh)));
My = [ My0, -1i*My0 ]; %#ok<MINV>
Md = [ Md0, -1i*Md0 ];

Ky = diag(vh)\real( diag(conj(Vh))*My );
Kd  = diag(vh)\real( diag(conj(Vh))*Md );

a = w;
b = vh - Ky*xhy - Kd*xhd;

% define linear model:
% vc = My*xhy + Md*xhd + a;
% vm = Ky*xhy + Kd*xhd + b;


end