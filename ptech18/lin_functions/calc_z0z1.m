function [Z1, Z0] = calc_z0z1( Asc3,Asc1,X1R1,X0R0,Vbase,type )

% NOMINALLY: Asc3 = 2000 Asc1 = 2100 X1R1 = 4 X0R0 = 3 type=mva?

switch type
    case 'mva' %NB S = VIsqrt(3)*1e-3
        absZ0 = (Vbase^2)/Asc3;
        absZph1 = (Vbase^2)/Asc1;
        Z0 = absZ0*exp(1i*atan(X0R0));
        Z1 = z1_3ph( Z0, absZph1, X1R1 );
    case 'i'
        absZ0 = Vbase/(Asc3*sqrt(3)*1e-3);
        absZph1 = Vbase/(Asc1*sqrt(3)*1e-3);
        Z0 = absZ0*exp(1i*atan(X0R0));
        Z1 = z1_3ph( Z0, absZph1, X1R1 );
    case 'mva1'
        Z1 = ((Vbase^2)/Asc1)*exp(1i*atan(X0R0));
        Z0 = NaN;
    case 'i1'
        Z1 = NaN;
        Z0 = NaN;
    case 'z1z0'
        Z1 = Asc3 + 1i*Asc1;
        Z0 = X1R1 + 1i*X0R0;
end


end

