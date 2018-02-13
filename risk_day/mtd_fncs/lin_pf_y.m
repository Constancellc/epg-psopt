function [ vc,Sfd,Sld,Sls ] = lin_pf_y( My,a,Ybus,v0,xhy )
% NB putting Ybus in as a sparse makes this much quicker!

vc = My*xhy + a;
Vlin = [v0;vc];
Slin = Vlin.*conj(Ybus*Vlin)/1e3;

Sfd = sum(Slin(1:3));
Sld = sum(Slin(4:end));
Sls = sum(Slin);


end

