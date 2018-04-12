function Z1 = z1_3ph( Z0, absZph1, X1R1 )
% see workbook 170817. Calculating Z1 (Zy) for an opendss vsource object,
% when the short circuit parameters are known.

R0 = real(Z0);
X0 = imag(Z0);

tht = atan(X1R1);
absZ1 = 0.5*(  -(R0*cos(tht) + X0*sin(tht)) + ...
                sqrt( (9*(absZph1^2)) - (( R0*sin(tht) - X0*cos(tht) )^2) ) );

Z1 = absZ1*exp(1i*tht);


end

