clear all;
set(0,'DefaultTextInterpreter','Latex')
% addpath('lin_functions');
%%
FD = 'C:\Users\Matt\Documents\DPhil\pesgm19\pesgm19_paper\tables\';
pltD = 'C:\Users\Matt\Documents\DPhil\papers\pesgm19\pesgm19_poster\figures\';
nl = 0.5;

thesisTable = 1;

modes = {'Wfix','Vfix'};
% modes = {'Vfix'};

frac_set = 0.05;

% models = {'eulv','n1f1','n2f1','n3f1','n4f1'};
models = {'eulv','n1f1','n2f1','n3f1','n4f1','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24','4busYy','epriK1cvr','epri24cvr','123busCvr'};
model_is = [4+1];
model_is = [20,17,18,22]+1;

Ns = 3000; % number of samples
tConv = 0.003; % convergence criteria
% qgen = 0;

mc_time = zeros(numel(model_is),numel(modes));
Pout = zeros(numel(model_is),numel(modes));
counts = zeros(numel(model_is),1);

for K = 1:numel(model_is)
    
    modeli = model_is(K);
    
    if modeli<6
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
    
    rng(0);
    model = models{modeli};
    display(model );
    
    load([pwd,'\lin_models\',model]);
    LDScount = nnz(xhy0)/2;
    Nl = ceil(nl*LDScount);
    sn = [pwd,'\lin_models\mc_out_',model];

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
    
    Ky = real((1./ang0).*My);
%     Ky2 = real(asd*My);
%     isequal(Ky,Ky2)
    Kynew = Ky(:,1:numel(xhy0)/2);
    bnew = b;
%     Kynew = Ky(xhp0~=0,1:numel(xhy0)/2);
%     bnew = b(xhp0~=0);


    nV = numel(bnew);
    nS = numel(xhy0);
    nP = numel(xhy0)/2;

    PPa = [];
    PPb = [];
    
    for k = 1:numel(modes)
        mode = modes{k};
        display(mode)
        
        tic
        if strcmp(mode,'Vfix')
            for i = 1:numel(Nl)
                for j = 1:Ns
                    rdi = randperm(LDScount,Nl(i));
                    xhs = sparse(fxp0(rdi),1,ones(Nl(i),1),nP,1);
                    Anew = Kynew*xhs;
                    xab = bnew./Anew;
                    xab = xab + 0./(xab>0); % get rid of negative values

                    X(j,i) = min(xab); % this is the value that maximises the o.f.#
                end
            end
            kX = X.*Nl*1e-3;
            Pout(K,k) = quantile(kX(:),frac_set);
            
        elseif strcmp(mode,'Wfix')
            xhs = sparse(fxp0,1,ones(numel(fxp0),1),nP,1);
            Anew = Kynew*xhs;
            
            xab = bnew./Anew;
            xab = xab + 0./(xab>0); % get rid of negative values
            
            P100 = min(xab); % this is the value that maximises the o.f.
            
            Pa = 0;
            frac_a = -frac_set;
            
            Pb = P100;
            for i = 1:numel(Nl)
                Pgen = Pb*LDScount/Nl(i);
                for j = 1:Ns
                    rdi = randperm(LDScount,Nl(i));
                    xhs = sparse(fxp0(rdi),1,ones(Nl(i),1)*Pgen,nP,1);
                    Vout = Va0 + Kynew*xhs; % Vout2

                    Vout_ld = Vout/Vb;
                    Vout_mx(j,i) = max(abs(Vout_ld));
                end
            end
            frac_b = (nnz(Vout_mx>vp)/numel(Vout_mx)) - frac_set;
            Pc = 0.5*(Pb + Pa);
            
            error = abs((frac_a-frac_b))/(abs(frac_b) + 1);
            count = 0;
            while error > tConv
                PPa(end+1) = Pa;
                PPb(end+1) = Pb;
                for i = 1:numel(Nl)
                    Pgen = Pc*LDScount/Nl(i);
                    for j = 1:Ns
                        rdi = randperm(LDScount,Nl(i));
                        xhs = sparse(fxp0(rdi),1,ones(Nl(i),1)*Pgen,nP,1);
                        Vout = Va0 + Kynew*xhs; % Vout2
                        Vout_ld = Vout/Vb;
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
            Pout(K,k) = mean([Pa,Pb])*LDScount*1e-3;
            counts(K) = count;
        end
        mc_time(K,k)=toc;
    end
end

%%
Tnm = 'montecarlo_comparison';
mat = [counts,mc_time(:,1),Pout(:,1),mc_time(:,2),Pout(:,2)];
Tfn = [FD,Tnm];
display(mat)

headerRow = {'Feeder','Iterations','Time, s','$\Phi _{5\%}$, kW','Time, s','$\Phi _{5\%}$, kW'};
headerCol = models(model_is);
caption = 'Comparison of timings and estimated hosting capacities for the fixed power and fixed voltage methods';
formatCol = {'$%d$','$%.2f$','$%.1f$','$%.2f$','$%.1f$'};

% matrix2latex(mat,Tfn,'label',Tnm,'formatColumns',formatCol,'alignment','l','caption',caption,...
%                     'headerRow',headerRow,'headerColumn',headerCol)

if exist('thesisTable')
    mat = [mc_time(:,1),Pout(:,1),mc_time(:,2),Pout(:,2),mc_time(:,1)./mc_time(:,2),counts];
    headerRow = {'Feeder','Time, s','$\Phi _{5\%}$, kW','Time, s','$\Phi _{5\%}$, kW','Time ratio','Iterations'};
    caption = 'Comparison of timings and estimated hosting capacities for the fixed power and fixed voltage methods';
    formatCol = {'$%.2f$','$%.1f$','$%.2f$','$%.1f$','$%.1f$','$%d$'};
    
    Tnm = 'montecarlo_comparison_thesis';
    TF = ['C:\Users\Matt\Documents\DPhil\thesis\c4tech2\c4tables\',Tnm];
    matrix2latex(mat,TF,'label',Tnm,'formatColumns',formatCol,'alignment','l','caption',caption,...
                    'headerRow',headerRow,'headerColumn',headerCol)
end

% %% POSTER BAR CHARTS
% figure('Position',[100 150 260 400]);
% subplot(211)
% bar(mc_time);
% set(gca,'yscale','log')
% ylim([0.06,300])
% yticks([0.1,1,10,100])
% % xticklabels({'EU LV','N1.1','N2.1','N3.1','N4.1'})
% xticklabels({'N1','N2','N3','N4','N5'})
% xlabel('Feeder ID')
% ylabel('Solution time, s')
% 
% subplot(212)
% bar(Pout);
% % set(gca,'yscale','log')
% xticklabels({'N1','N2','N3','N4','N5'})
% xlabel('Feeder ID')
% ylabel('Hosting Capacity $\Phi_{5\%}$, kW')
% ylim([0,400])
% legend('Fixed power','Fixed voltage')
% 
% FL = [pltD,'monteCarloComparisonPstr'];
% export_fig(gcf,[FL,'.pdf'],'-pdf','-transparent');














                    