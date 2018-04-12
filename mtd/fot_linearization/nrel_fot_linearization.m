function [ My,Md,a,Ky,Kd,b ] = nrel_fot_linearization( xh,H,Y,Vh,V0,iD,sD )

%------------
Yll = Y(4:end,4:end);
Yl0 = Y(4:end,1:3);

Ny = size(H,2);
xhy = xh(1:2*Ny);
xhd = xh(2*Ny+1:end);

% iDc = conj(iD(4:end));
iDc = diag(H*Vh)\sD(4:end)*1e3;
U = [eye(numel(Vh)), 1i*eye(numel(Vh))];
ZRO = zeros(size(U));

%----------
% 15a
vxy_a = diag( H'*iDc - conj(Yl0*V0 + Yll*Vh) );
vxyc_a = -diag(Vh)*conj(Yll);
iDcxy_a = diag(Vh)*H';
c_a = -U;

% 15b
vxy_b = -diag(iDc)*H;
iDcxy_b = -diag(H*Vh);
c_b = ZRO;

% 15c
vxd_c = diag( H'*iDc - conj(Yl0*V0 + Yll*Vh) );
vxdc_c = -diag(Vh)*conj(Yll);
idcxd_c = diag(Vh)*H';
c_c = ZRO;

% 15d
vxd_d = -diag(iDc)*H;
iDcxd_c = -diag(H*Vh);
c_d = -U;
%----------
zro = zeros(size(bracket(vxy_a)));

AA = [ 
      [ bracket(vxy_a) + bracket_c(vxyc_a) ,bracket_c(iDcxy_a), zro,zro ];
      [ bracket(vxy_b)                     ,bracket_c(iDcxy_b), zro,zro ];
      [ zro,zro, bracket(vxd_c) + bracket_c(vxdc_c),bracket_c(idcxd_c) ];
      [ zro,zro, bracket(vxd_d)                    ,bracket_c(iDcxd_c) ];
      ];
       
BB = [real(c_a);imag(c_a);
      real(c_b);imag(c_b);
      real(c_c);imag(c_c);
      real(c_d);imag(c_d)];

XX = AA\BB;
%----------

My = XX(1:Ny,:) + 1i*XX(Ny+1:Ny*2,:);
Md = XX(4*Ny+1:5*Ny,:) + 1i*XX(5*Ny+1:Ny*6,:);

Ky = diag(Vh)\real( diag(conj(Vh))*My );
Kd  = diag(Vh)\real( diag(conj(Vh))*Md );


a = Vh - My*xhy - Md*xhd;
b = Vh - Ky*xhy - Kd*xhd;

end