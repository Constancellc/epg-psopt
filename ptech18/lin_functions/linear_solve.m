function [ Vg,Xnut ] = linear_solve( Amat,Sg_mes,BB0,nut_idx,Pg,vb )
% NB the units of Amat and Sg_mes must match (either both pu or both actual
% values!)
% Pg is a variable simply included for reshaping matrices

Pg_mes = real(Sg_mes);
Qg_mes = imag(Sg_mes);
n = numel(BB0)/4;
n_nut = numel(nut_idx);

X = zeros(4*n,numel(Pg));

for i = 1:numel(Pg)
    BB = BB0;
    
    idx_p = (2*n + 3) + nut_idx;
    idx_q = (3*n) + nut_idx;
    
    BB(idx_p) = BB0(idx_p) - (Pg_mes(i,:)/n_nut);
    BB(idx_q) = BB0(idx_q) - (Qg_mes(i,:)/n_nut);
    
    X(:,i) = Amat\BB;
end

Xnut = X(nut_idx,:)/(vb*sqrt(1/3)*1e3);
XV = mean(Xnut,1);
Vg = reshape(XV,size(Pg));



end

