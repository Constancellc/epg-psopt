clear all;
set(0,'DefaultTextInterpreter','Latex')
% addpath('lin_functions');

FD = 'C:\Users\Matt\Documents\DPhil\pesgm19\pesgm19_paper\tables\';
nl = 0.5;

modes = {'Wfix','Vfix'};


frac_set = 0.05;

models = {'eulv','n1f1','n2f1','n3f1','n4f1'};
% models = {'eulv'};
fn{1} = [pwd,'\LVTestCase_copy\master_z_g'];
fn{2} = [pwd,'\manchester_models\network_1\Feeder_1\master_g'];
fn{3} = [pwd,'\manchester_models\network_2\Feeder_1\master_g'];
fn{4} = [pwd,'\manchester_models\network_3\Feeder_1\master_g'];
fn{5} = [pwd,'\manchester_models\network_4\Feeder_1\master_g'];

[~,DSSObj,DSSText] = DSSStartup;
DSSCircuit = DSSObj.ActiveCircuit;
DSSSolution = DSSCircuit.Solution;
LDS = DSSCircuit.loads;

Ns = 1000; % number of samples

Vb = 230;
vp = 1.10;
vmax = Vb*vp;
% gen_pf = -0.95;
gen_pf = 1.00;
aa = exp(1i*2*pi/3);

qgen = sin(acos(abs(gen_pf)))/gen_pf;

mc_time = zeros(numel(models),numel(modes));
Pout = zeros(numel(models),numel(modes));
counts = zeros(numel(models),1);


% mode = 'Wfix'; K = 5;
for k = 1:numel(modes)
    mode = modes{k};
    for K = 1:numel(models)
        display(K);
        rng(0);
        modeli = K;
        
        DSSText.command=['Compile (',fn{modeli},')'];
        model = models{modeli};

        load([pwd,'\lin_models\',model]);
        Nl = ceil(nl*LDS.count);
        sn = [pwd,'\lin_models\mc_out_',model];
        % sn = [pwd,'\lin_models\mc_out_',model,'_upf']

        vmag=zeros(Ns,numel(Nl));
        X=zeros(Ns,numel(Nl));
        Vub=zeros(Ns,numel(Nl));
        Vout_mx=zeros(Ns,numel(Nl));

        tic

        % set up linear model & solve:
        f = -1;
        xhy0=sparse(xhy0);
        Vh0 = My*xhy0 + a;
        ang0 = exp(1i*angle(Vh0));
        Va0 = real(Vh0./ang0);

        Vmax = vmax*ones(size(Vh0));
        b = Vmax - Va0;

        xhp0 = xhy0(1:numel(xhy0)/2);
        fxp0 = find(xhp0);

        PPa = [];
        PPb = [];
        if strcmp(mode,'Vfix')
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
            Pout(K,k) = quantile(kX(:),frac_set);
            
        elseif strcmp(mode,'Wfix')
            xhs = sparse(zeros(size(xhp0)));
            xhs(fxp0) = 1;
            xhs = [xhs;xhs*qgen];
            Mk = My*xhs;
            A = real(Mk./ang0);

            Anew = A(xhp0~=0);
            bnew = b(xhp0~=0);
            
            xab = bnew./Anew;
            xab = xab + 0./(xab>0); % get rid of negative values
            
            P100 = min(xab); % this is the value that maximises the o.f.
            
            Pa = 0;
            frac_a = -frac_set;
            
            Pb = P100;
            for i = 1:numel(Nl)
                for j = 1:Ns
                    rdi = randperm(LDS.count,Nl(i));
                    xhs = sparse(zeros(size(xhp0)));
                    xhs(fxp0(rdi)) = Pb*LDS.count/Nl(i);
                    xhs = [xhs;xhs*qgen];
                    
                    Vout = My*(xhy0 + xhs) + a;
                    Vout_ld = Vout(xhp0~=0)/Vb;
                    Vout_mx(j,i) = max(abs(Vout_ld));
                end
            end
            frac_b = (nnz(Vout_mx>vp)/numel(Vout_mx)) - frac_set;
            Pc = 0.5*(Pb + Pa);
            
            error = abs((frac_a-frac_b))/(abs(frac_b) + 1);
            count = 0;
            while error > 0.01
                PPa(end+1) = Pa;
                PPb(end+1) = Pb;
                for i = 1:numel(Nl)
                    for j = 1:Ns
                        rdi = randperm(LDS.count,Nl(i));
                        xhs = sparse(zeros(size(xhp0)));
                        xhs(fxp0(rdi)) = Pc*LDS.count/Nl(i);
                        xhs = [xhs;xhs*qgen];
                        
                        Vout = My*(xhy0 + xhs) + a;
                        Vout_ld = Vout(xhp0~=0)/Vb;
                        Vout_mx(j,i) = max(abs(Vout_ld));
                    end
                end
                
                frac_c = (nnz(Vout_mx>vp)/numel(Vout_mx)) - frac_set;
                
                if frac_c > 0
                    Pa = Pa; frac_a = frac_a;
                    Pb = Pc; frac_b = frac_c;
                    Pc = 0.5*(Pa + Pb);
                else
                    Pa = Pc; frac_a = frac_c;
                    Pb = Pb; frac_b = frac_b;
                    Pc = 0.5*(Pa + Pb);
                end
                error = abs((frac_a-frac_b))/(abs(frac_b) + 1);
                count = count + 1;
            end
            Pout(K,k) = mean([Pa,Pb])*LDS.count*1e-3;
            counts(K) = count;
        end
        mc_time(K,k)=toc;
    end
end


Tnm = 'montecarlo_comparison';
mat = [counts,mc_time(:,1),Pout(:,1),mc_time(:,2),Pout(:,2)];
Tfn = [FD,Tnm];

headerRow = {'Feeder','Iterations','Time, s','$\Phi _{5\%}$, kW','Time, s','$\Phi _{5\%}$, kW'};
headerCol = {'EU LV','N1.1','N2.1','N3.1','N4.1'};
caption = 'Comparison of timings and estimated hosting capacities for the fixed power and fixed voltage methods'
formatCol = {'$%d$','$%.2f$','$%.1f$','$%.2f$','$%.1f$'};

matrix2latex(mat,Tfn,'label',Tnm,'formatColumns',formatCol,'alignment','l','caption',caption,...
                    'headerRow',headerRow,'headerColumn',headerCol)



                    