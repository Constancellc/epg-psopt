clear all; % close all;
tic
set(0,'DefaultTextInterpreter','Latex')
% addpath('lin_functions');

FD = ['C:\Users\',getenv('username'),'\Documents\DPhil\pesgm19\pesgm19_paper\tables\'];

% tableExport = 1;
% mc_validation = 1;
% montecarlo_validation_thesis = 1;

models = {'eulv','n1f1','n2f1','n3f1','n4f1','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24','4busYy','epriK1cvr','epri24cvr','123busCvr'};
modelSht = {'eulv','n1f1','n2f1','n3f1','n4f1','13bus','34bus','37bus','123bus','8500node','epri5','epri7','epriJ1','epriK1','epriM1','epri24'};
modelShtTidy = {'EU LV','N1.F1','N2.F1','N3.F1','N4.F1','13 Bus','34 Bus','37 Bus','123 Bus','8500 Node','Ckt. 5','Ckt. 7','Ckt. J1','Ckt. K1','Ckt. M1','Ckt. 24'};
modelsTidy = containers.Map(modelSht,modelShtTidy);

model_is = [9,22,20,0,1,2,3,4]+1;

nl = 0.5;
frac_set = 0.05;

Ns = 2*1000; % number of samples

Pout = zeros(numel(model_is),2);
for K = 1:numel(model_is)
    modeli = model_is(K);
    display(models{modeli})
    rng(0); % for repeatability
    
    if and(modeli>1,modeli<6)
        Vb = 230;
        vp = 1.10;
    else
        Vb = 1;
        if modeli==1
            vp=1.10;
        else
            vp = 1.055;
        end
    end
    vmax = Vb*vp;
    
    
    
%     DSSText.Command=['Compile (',fn{modeli},')'];
    model = models{modeli};

    load([pwd,'\lin_models\',model]);
    LDScount = nnz(xhy0)/2;
    Nl = ceil(nl*LDScount);
%     Nl = ceil(nl*LDS.count);

    sn = [pwd,'\lin_models\mc_out_',model];

    vmag=zeros(Ns,numel(Nl));
    X=zeros(Ns,numel(Nl));
    Vub=zeros(Ns,numel(Nl));
    Vout_mx=zeros(Ns,numel(Nl));

    % set up linear model & solve:
    xhy0=sparse(xhy0);
    Vh0 = My*xhy0 + a;
    ang0 = exp(1i*angle(Vh0));
    Va0 = real(Vh0./ang0);

    Vmax = vmax*ones(size(Vh0));
    b = Vmax - Va0;

    xhp0 = xhy0(1:numel(xhy0)/2);
    fxp0 = find(xhp0);

    nV = numel(b);
    nS = numel(xhy0);
    nP = numel(xhy0)/2;

    Ky = real((1./ang0).*My);
    Kynew = Ky(:,1:numel(xhy0)/2);
    
    xhs = [];
    for j = 1:Ns
        rdi = randperm(LDScount,Nl);
        xhs = [xhs,sparse(fxp0(rdi),1,ones(Nl,1),nP,1)];
    end
    
    AnewSet = Kynew*xhs;
    for j = 1:Ns
        Anew = AnewSet(:,j);
        xab = b./Anew;
        X(j) = min(xab(xab>0)); % this is the value that maximises the o.f.#
    end
    kX = X.*Nl*1e-3;
%     Pout(K,k) = quantile(kX(:),frac_set);
%     PoutGen(K,k) = quantile(X(:),frac_set)*1e-3;
    
    
%     for j = 1:Ns
%         rdi = randperm(LDScount,Nl(i));
%         xhs = sparse(zeros(size(xhp0)));
% 
%         xhs(fxp0(rdi)) = 1;
%         xhs = [xhs;xhs*qgen];
%         Mk = My*xhs;
%         A = real(Mk./ang0);
%         Anew = A(xhp0~=0);
%         bnew = b(xhp0~=0);
% 
%         xab = bnew./Anew;
%         xab = xab + 0./(xab>0); % get rid of negative values
% 
%         X(j,i) = min(xab); % this is the value that maximises the o.f.
%         Vout = My*(xhy0 + xhs*X(j,i)) + a;
%     end
%     kX = X.*Nl*1e-3;

    Pout(K,1) = quantile(kX(1:Ns/2),frac_set);
    Pout(K,2) = quantile(kX((Ns/2)+1:end),frac_set);
    PoutGen(K,1) = 1e-3*quantile(X(1:Ns/2),frac_set);
    PoutGen(K,2) = 1e-3*quantile(X((Ns/2)+1:end),frac_set);

end
display(Pout);

Pout_frac = 100*abs(diff(Pout')')./abs(Pout(:,1));
PoutGen_frac = 100*abs(diff(PoutGen')')./abs(PoutGen(:,1));

if exist('mc_validation','var')
    Tnm = 'montecarlo_validation';
    mat = [Pout,Pout_frac];
    Tfn = [FD,Tnm];

    headerRow = {'Feeder','$\Phi_{5\%}$ (run A), kW','$\Phi_{5\%}$ (run B), kW','Rel. error, \%'};
    headerCol = {'EU LV','N1.1','N2.1','N3.1','N4.1'};
    caption = 'Estimated hosting capacity and error for two monte carlo runs ($N_{\mathrm{MC}}$ = 1000, $n_{\mathrm{pen}}=50\%$)';
    formatCol = {'$%.1f$','$%.1f$','$%.2f$'};

    if exist('tableExport;','var')
        matrix2latex(mat,Tfn,'label',Tnm,'formatColumns',formatCol,'alignment','l','caption',caption,...
                            'headerRow',headerRow,'headerColumn',headerCol)
    end                    
end                    
                    
%%                    
if exist('montecarlo_validation_thesis','var')
    Tnm = 'montecarlo_validation_thesis';
    mat = [PoutGen,PoutGen_frac];

    headerRow = {'Feeder','$P_{\mathrm{Gen}}^{5\%}$ (i), kW','$P_{\mathrm{Gen}}^{5\%}$ (ii), kW','Rel. error, \%'};
    headerCol = {};
    for ii = 1:numel(model_is)
        model = models(model_is(ii));
        headerCol{ii} = modelsTidy(model{:});
    end
    
    caption = 'Estimated hosting capacity and error for two monte carlo runs ($N_{\mathrm{MC}}$ = 1000, $\alpha_{\mathrm{PV}}=50\%$)';
    formatCol = {'$%.2f$','$%.2f$','$%.2f$'};
    
    Tfn = ['C:\Users\',getenv('username'),'\Documents\DPhil\thesis\c4tech2\c4tables\',Tnm];
    display(mat)
    if exist('tableExport','var')
        matrix2latex(mat,Tfn,'label',Tnm,'formatColumns',formatCol,'alignment','l','caption',caption,...
                            'headerRow',headerRow,'headerColumn',headerCol)
    end                    
end                  
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    