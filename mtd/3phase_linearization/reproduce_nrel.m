% reproduce_nrel is a script that looks to reproduce the results of the
% paper "load-flow in multiphase distrbution networks: existence,
% uniqueness, and linear models", available at
% https://arxiv.org/abs/1702.03310

% --------------------------
% FPL method (section IV.b), 37 node test feeder

% we need to get the appropriate matrices (Y,Vh,H,xh)



% find derived values
Nd = size(H,1);
Ny = size(H,2);

xhy = xh(1:Ny);
xhd = xh(Ny+1:end);

V0 = Vh(1:3);

Yll = Y(4:end,4:end);
Yl0 = Y(4:end,1:3);

w = -Yll\Yl0*V0;

% now create linearisation matrices
My0 = inv(diag(conj(Vh))*Yll);
Md0 = Yll\(H')/diag(conj(H*Vh));
My = [ My0 , -1i*My0 ]; %#ok<MINV>
Md = [ Md0 , -1i*Md0 ];

Ky = diag(abs(vh))\real( diag(conj(vh))*My );
Kd = diag(abs(vh))\real( diag(conj(vh))*Md );

a = w;
b = abs(vh) - Ky*xhy - kd*xhd;

% define linear model:
vc = My*xy + Md*xd + a;
vm = Ky*xy + kd*xd + b;