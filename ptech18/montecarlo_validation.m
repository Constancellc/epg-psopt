clear all; % close all;
tic
set(0,'DefaultTextInterpreter','Latex')
% addpath('lin_functions');

FD = 'C:\Users\Matt\Documents\DPhil\pesgm19\pesgm19_paper\tables\';

models = {'eulv','n1f1','n2f1','n3f1','n4f1'};
fn{1} = [pwd,'\LVTestCase_copy\master_z_g'];
fn{2} = [pwd,'\manchester_models\network_1\Feeder_1\master_g'];
fn{3} = [pwd,'\manchester_models\network_2\Feeder_1\master_g'];
fn{4} = [pwd,'\manchester_models\network_3\Feeder_1\master_g'];
fn{5} = [pwd,'\manchester_models\network_4\Feeder_1\master_g'];

nl = 0.5;
frac_set = 0.05;

[~,DSSObj,DSSText] = DSSStartup;
DSSCircuit = DSSObj.ActiveCircuit;
DSSSolution = DSSCircuit.Solution;
LDS = DSSCircuit.loads;

Vb = 230;
vp = 1.10;
vmax = Vb*vp;
gen_pf = 1.00;
aa = exp(1i*2*pi/3);

Ns = 2*1000; % number of samples

qgen = sin(acos(abs(gen_pf)))/gen_pf;

Pout = zeros(numel(fn),2);

for K = 1:numel(fn)
    modeli = K
    rng(0); % for repeatability

    DSSText.command=['Compile (',fn{modeli},')'];
    model = models{modeli};

    load([pwd,'\lin_models\',model]);
    Nl = ceil(nl*LDS.count);

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

    for i = 1:numel(Nl)
        for j = 1:Ns
            rdi = randperm(LDS.count,Nl(i));
            xhs = sparse(zeros(size(xhp0)));

            xhs(fxp0(rdi)) = 1;
            xhs = [xhs;xhs*qgen];
            Mk = My*xhs;
            A = real(Mk./ang0);
            Anew = A(xhp0~=0);
            bnew = b(xhp0~=0);

            xab = bnew./Anew;
            xab = xab + 0./(xab>0); % get rid of negative values

            X(j,i) = min(xab); % this is the value that maximises the o.f.
            Vout = My*(xhy0 + xhs*X(j,i)) + a;
        end
    end
    kX = X.*Nl*1e-3;
    Pout(K,1) = quantile(kX(1:Ns/2),frac_set);
    Pout(K,2) = quantile(kX((Ns/2)+1:end),frac_set);

end
display(Pout);

Pout_frac = 100*abs(diff(Pout')')./abs(Pout(:,1))

Tnm = 'montecarlo_validation';
mat = [Pout,Pout_frac];
Tfn = [FD,Tnm];

headerRow = {'Feeder','$\Phi_{5\%}$ (run A), kW','$\Phi_{5\%}$ (run B), kW','Rel. error, \%'};
headerCol = {'EU LV','N1.1','N2.1','N3.1','N4.1'};
caption = 'Estimated hosting capacity and error for two monte carlo runs ($n$ = 1000, $N_{\mathrm{Pen}}=50\%$)';
formatCol = {'$%.1f$','$%.1f$','$%.2f$'}

matrix2latex(mat,Tfn,'label',Tnm,'formatColumns',formatCol,'alignment','l','caption',caption,...
                    'headerRow',headerRow,'headerColumn',headerCol)
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    