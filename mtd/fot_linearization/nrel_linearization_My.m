function [ My,a ] = nrel_linearization_My( Y_Sp,Vh,V0 )
% put Ybus in as a sparse matrix for speedier running
Yll = sparse(Y_Sp(4:end,4:end));
Yl0 = sparse(Y_Sp(4:end,1:3));

w = -Yll\Yl0*V0;
a = w;

My0 = inv(diag(conj(Vh))*Yll);
My = [ My0, -1i*My0 ]; %#ok<MINV>


end