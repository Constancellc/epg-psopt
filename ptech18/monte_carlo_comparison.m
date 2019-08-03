clear all;
set(0,'DefaultTextInterpreter','Latex')
% addpath('lin_functions');
%%
FD = 'C:\Users\Matt\Documents\DPhil\pesgm19\pesgm19_paper\tables\';
pltD = 'C:\Users\Matt\Documents\DPhil\papers\pesgm19\pesgm19_poster\figures\';
nl = 0.5;

% paperTable = 1;
% montecarlo_comparison_thesis = 1;
% tableExport = 1;

% modes = {'Wfix','Vfix','Wscan'};
modes = {'Vfix','Wfix'};

frac_set = 0.05;

% models = {'eulv','n1f1','n2f1','n3f1','n4f1'};
models = {'eulv','n1f1','n2f1','n3f1','n4f1','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24','4busYy','epriK1cvr','epri24cvr','123busCvr'};
model_is = [9,22,20,0,1,2,3,4]+1;
modelSht = {'eulv','n1f1','n2f1','n3f1','n4f1','13bus','34bus','37bus','123bus','8500node','epri5','epri7','epriJ1','epriK1','epriM1','epri24'};
modelShtTidy = {'EU LV','N1.F1','N2.F1','N3.F1','N4.F1','13 Bus','34 Bus','37 Bus','123 Bus','8500 Node','Ckt. 5','Ckt. 7','Ckt. J1','Ckt. K1','Ckt. M1','Ckt. 24'};
modelsTidy = containers.Map(modelSht,modelShtTidy);
% model_is = [1,2,5];
% model_is = [20,17,18,22]+1;

Ns = 1000; % number of samples
tConv = 0.001; % convergence criteria
% qgen = 0;

mc_time = zeros(numel(model_is),numel(modes));
Pout = zeros(numel(model_is),numel(modes));
PoutGen = zeros(numel(model_is),numel(modes));
counts = zeros(numel(model_is),1);
ldsCountSet = zeros(numel(model_is),1);

for K = 1:numel(model_is)
    
    modeli = model_is(K);
    

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
    
    rng(0);
    model = models{modeli};
    display(model );
    
    load([pwd,'\lin_models\',model]);
    LDScount = nnz(xhy0)/2;
    Nl = ceil(nl*LDScount);
    ldsCountSet(K) = LDScount;
    sn = [pwd,'\lin_models\mc_out_',model];

    vmag=zeros(Ns,1);
    X=zeros(Ns,1);
    Vub=zeros(Ns,1);
    Vout_mx=zeros(Ns,1);
    
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
    Kynew = Ky(:,1:numel(xhy0)/2);

    nV = numel(b);
    nS = numel(xhy0);
    nP = numel(xhy0)/2;

    xhs = [];
    for j = 1:Ns
        rdi = randperm(LDScount,Nl);
        xhs = [xhs,sparse(fxp0(rdi),1,ones(Nl,1),nP,1)];
    end
    
    PPa = [];
    PPb = [];
    
    for k = 1:numel(modes)
        mode = modes{k};
        display(mode)
        tic
        if strcmp(mode,'Vfix')
%             xhs = [];rng(0);
%             for j = 1:Ns
%                 rdi = randperm(LDScount,Nl);
%                 xhs = [xhs,sparse(fxp0(rdi),1,ones(Nl,1),nP,1)];
%             end
    
            AnewSet = Kynew*xhs;
            for j = 1:Ns
                Anew = AnewSet(:,j);
                xab = b./Anew;
                X(j) = min(xab(xab>0)); % this is the value that maximises the o.f.#
            end
            kX = X.*Nl*1e-3;
            Pout(K,k) = quantile(kX(:),frac_set);
            PoutGen(K,k) = quantile(X(:),frac_set)*1e-3;
            
        elseif strcmp(mode,'Wscan')
            xhs100 = sparse(fxp0,1,ones(numel(fxp0),1),nP,1);
            Anew = Kynew*xhs100;

            xab = b./Anew;
            xab = xab + 0./(xab>0); % get rid of negative values

            P100 = min(xab); % this is the value that maximises the o.f.

            
            
            Pa = 0;
            frac_a = -frac_set;
            
            Pb = P100;
            Pgen = Pb*LDScount/Nl;
            
%             toScale = Kynew*xhs;
            
            nPgen = 3e2;
            PgenSet = linspace(0,Pgen,nPgen);
            fracOut = zeros(nPgen,1);
            for i = 1:nPgen
%                 fracOut(i) = (nnz(Vout_mx>vp)/numel(Vout_mx));
                for j = 1:Ns
                    Vout = Va0 + toScale(:,j)*PgenSet(i); % Vout2
                    Vout_mx(j) = max(Vout)/Vb;
                end
                fracOut(i) = (nnz(Vout_mx>vp)/numel(Vout_mx));
            end
            [~,iMin] = min(abs(fracOut-frac_set));
            Pout(K,k) = PgenSet(iMin)*Nl*1e-3;
            PoutGen(K,k) = PgenSet(iMin)*1e-3;
            display((PgenSet(2) - PgenSet(1))*Nl*1e-3)
            plot(PgenSet,fracOut)
            
        elseif strcmp(mode,'Wfix')
            xhs100 = sparse(fxp0,1,ones(numel(fxp0),1),nP,1);
            Anew = Kynew*xhs100;

            xab = b./Anew;
            P100 = min(xab(xab>0)); % this is the value that maximises the o.f.

            Pa = 0;
            frac_a = -frac_set;
            
            Pb = P100;
            Pgen = Pb*LDScount/Nl;
            
%             xhs = [];rng(0);
%             for j = 1:Ns
%                 rdi = randperm(LDScount,Nl);
%                 xhs = [xhs,sparse(fxp0(rdi),1,ones(Nl,1),nP,1)];
%             end

            toScale = Kynew*xhs;
            
            for j = 1:Ns
                Vout = Va0 + toScale(:,j)*Pgen; % Vout2
                Vout_mx(j) = max(Vout)/Vb;
            end
            frac_b = (nnz(Vout_mx>vp)/numel(Vout_mx)) - frac_set;
            Pc = 0.5*(Pb + Pa);
            
            error = abs((frac_a-frac_b))/(abs(frac_b) + 1);
            count = 0;
            
%             scale0 = 
            
            while error > tConv
                PPa(end+1) = Pa;
                PPb(end+1) = Pb;
                Pgen = Pc*LDScount/Nl;
                
%                 xhs = [];rng(0);
%                 for j = 1:Ns
%                     rdi = randperm(LDScount,Nl);
%                     xhs = [xhs,sparse(fxp0(rdi),1,ones(Nl,1),nP,1)];
%                 end

                toScale = Kynew*xhs;
                for j = 1:Ns
                    Vout = Va0 + toScale(:,j)*Pgen; % Vout2
                    Vout_mx(j) = max(Vout)/Vb;
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
            PoutGen(K,k) = mean([Pa,Pb])*LDScount*1e-3/Nl;
            counts(K) = count;
        end
        mc_time(K,k)=toc;
    end
end
display(mc_time)
display(count)
% V0 = Kynew*xhs;
% 
% kMult = 1;
% Vact = kMult*V0 + Va0;

%%

headerCol = models(model_is);
caption = 'Comparison of timings and estimated hosting capacities for the fixed power and fixed voltage methods';

if exist('paperTable','var')
    Tnm = 'montecarlo_comparison';
    mat = [counts,mc_time(:,1),Pout(:,1),mc_time(:,2),Pout(:,2)];
    Tfn = [FD,Tnm];
    display(mat)

    headerRow = {'Feeder','Iterations','Time, s','$\Phi _{5\%}$, kW','Time, s','$\Phi _{5\%}$, kW'};
    formatCol = {'$%d$','$%.2f$','$%.1f$','$%.2f$','$%.1f$'};


    if exist('tableExport','var')
        matrix2latex(mat,Tfn,'label',Tnm,'formatColumns',formatCol,'alignment','l','caption',caption,...
                        'headerRow',headerRow,'headerColumn',headerCol)
    end
end
%%
if exist('montecarlo_comparison_thesis','var')
    mat = [mc_time(:,1),PoutGen(:,1),mc_time(:,2),PoutGen(:,2),counts,mc_time(:,2)./mc_time(:,1)];
    headerRow = {'Feeder','Time, s','$P_{\mathrm{Gen}}^{5\%}$, kW','Time, s','$P_{\mathrm{Gen}}^{5\%}$, kW','Iterations','Time ratio'};
    caption = 'Comparison of timings and estimated hosting capacities for the fixed power and fixed voltage methods';
    formatCol = {'$%.2f$','$%.3f$','$%.2f$','$%.3f$','$%d$','$%.1f$'};
    
    headerCol = {};
    for ii = 1:numel(model_is)
        model = models(model_is(ii));
        headerCol{ii} = modelsTidy(model{:});
    end
    
    
    Tnm = 'montecarlo_comparison_thesis';
    TF = ['C:\Users\',getenv('username'),'\Documents\DPhil\thesis\c4tech2\c4tables\',Tnm];
    display(mat)
    if exist('tableExport','var')
        matrix2latex(mat,TF,'label',Tnm,'formatColumns',formatCol,'alignment','l','caption',caption,...
                    'headerRow',headerRow,'headerColumn',headerCol)
    end
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