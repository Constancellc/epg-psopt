function [ Z1,Z0 ] = source_impedance( vll,Isc3,Isc1,x1r1,x0r0 )

% assume for now sc3, sc1 are Isc
vln = vll/sqrt(3); % kV

z1_a = vln*1e3/Isc3;
zs_a = vln*1e3/Isc1;

Z1 = z1_a*exp(1i*atan(x1r1));
R1 = real(Z1); X1 = imag(Z1);


a = 1 + x0r0^2;
b = 4*R1 + 4*X1*x0r0;
c = 4*(R1^2 + X1^2) - 9*(zs_a^2);

RTS = roots([a b c]);

R0 = max(RTS);
X0 = x0r0*R0;

Z0 = R0+1i*X0;

end

