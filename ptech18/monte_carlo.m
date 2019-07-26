clear all; % close all;
tic

set(0,'DefaultTextInterpreter','Latex');

ax = gca; colorSet = ax.ColorOrder; close;
colorOrder = get(0, 'DefaultAxesColorOrder');

% paperPlt = 1
% posterPlt = 1;
% thesisPlot = 1;
% plotExport = 1;
% thesisCasesTable = 1;
% tableExport = 1;

% FD = 'C:\Users\Matt\Documents\DPhil\pesgm19\pesgm19_paper\figures\'; % paper version
FD = ['C:\Users\',getenv('username'),'\Documents\DPhil\papers\pesgm19\pesgm19_poster\figures\']; % poster version


models = {'eulv','n1f1','n2f1','n3f1','n4f1','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24','4busYy','epriK1cvr','epri24cvr','123busCvr'};
modelSht = {'eulv','n1f1','n2f1','n3f1','n4f1','13bus','34bus','37bus','123bus','8500node','epri5','epri7','epriJ1','epriK1','epriM1','epri24'};
modelShtTidy = {'EU LV','N1.F1','N2.F1','N3.F1','N4.F1','13 Bus','34 Bus','37 Bus','123 Bus','8500 Node','Ckt. 5','Ckt. 7','Ckt. J1','Ckt. K1','Ckt. M1','Ckt. 24'};
modelsTidy = containers.Map(modelSht,modelShtTidy);

model_is = [0,20,9] + 1;
% model_is = [1,21];

nl = linspace(0,1,13);
nl(1) = 1e-4;
frac_set = 0.05;
Ns = 3000; % number of samples

caseStudies = zeros(numel(model_is),6);

for ii = 1:numel(model_is)
    modeli = model_is(ii);
    model = models{modeli};

    load([pwd,'\lin_models\',model]);
    LDScount = nnz(xhy0)/2;
    Nl = ceil(nl*LDScount);

    rng(0);

    if and(2<modeli,modeli<6)
%     if modeli<6
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

    vmag=zeros(Ns,numel(Nl));
    X=zeros(Ns,numel(Nl));

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
    Kynew = Ky(:,1:numel(xhy0)/2);

    nV = numel(b);
    nP = numel(xhy0)/2;

    tic
    % run Vfix
    for i = 1:numel(Nl)
        toc
        for j = 1:Ns
            rdi = randperm(LDScount,Nl(i));
            xhs = sparse(fxp0(rdi),1,ones(Nl(i),1),nP,1); % this does seem faster than non-sparse alternatives.
            Anew = Kynew*xhs;
            xab = b./Anew;
            xab = xab + 0./(xab>0); % get rid of negative values
            X(j,i) = min(xab); % this is the value that maximises the o.f.
        end
        Pout(i) = quantile(X(:,i),frac_set);
    end
    
    P100 = X(1,end);
    [~,i15] = min(abs(nl-0.15));
    [~,i40] = min(abs(nl-0.40));
    [~,i75] = min(abs(nl-0.75));
    caseStudies(ii,1) = 1e-3*P100;
    caseStudies(ii,2:4) = quantile(1e-3*X(:,i75),[0.01,0.05,0.10]);
    Xnew = sort(reshape(X(:,i15:i40),1,[]));
    
    caseStudies(ii,5) = 1e-3*quantile(Xnew,0.05);
    caseStudies(ii,6) = 1e-3*quantile(X(:,i40),0.01);
    
%     idxs = find(Xnew==P100);
%     caseStudies(ii,7) = 100*mean(idxs(1),idxs(end))/numel(Xnew);
    
    kX = X.*Nl*1e-3;
    FL = [FD,'variable_N_',num2str(modeli)];

    xx = 100*Nl./Nl(end);
    xtl = cellstr(num2str(xx','%.0f'));
    xtl{1} = num2str(xx(1),'%.2f');
    xtl(2:3:end) = {''};
    xtl(3:3:end) = {''};
    fdr = num2str(modeli);
    

    

    if exist('paperPlt','var')
        % xtl = cellstr(num2str(Nl'));
        fig = figure('Color','White','Position',[100 150 550 270]);
        subplot(121);
        plot(xx,Pout*1e-3.*Nl,'rx'); hold on;
        boxplot(kX,'Positions',xx,'Whisker',10); hold on;
        xticklabels(xtl);
        xlabel(['Penetration $n_{\mathrm{pen}}$, \% (Fdr. ',fdr,')']); 
        ylabel('Hosting capacity $\Phi$, kW'); grid on;
        xs = axis; axis([xs(1:2),0,xs(4)])
        lgnd = legend('$\Phi_{\mathrm{5\%}}$');
        set(lgnd,'Interpreter','Latex','FontSize',12,'Location','SouthEast');


        subplot(122);
        plot(xx,Pout*1e-3,'ro'); hold on;
        boxplot(X*1e-3,'Positions',xx,'Whisker',10); hold on;
        xticklabels(xtl)
        xlabel(['Penetration $n_{\mathrm{pen}}$, \% (Fdr. ',fdr,')']); 
        ylabel('Power per generator $\phi$, kW'); grid on;
        if modeli==3
            xs=axis; axis([xs(1:2),0,25]);
        elseif model==5
            xs = axis; axis([xs(1:2),0,40]);
        else
            xs = axis; axis([xs(1:2),0,xs(4)]);
        end

        lgnd = legend('$\phi_{\mathrm{5\%}}$');
        set(lgnd,'Interpreter','Latex','FontSize',12);

        export_fig(gcf,FL);
        export_fig(gcf,[FL,'.pdf'],'-pdf');
        saveas(gcf,FL,'meta')
    end

    if exist('posterPlt','var')
        close
        fig = figure('Position',[100 150 300 450]);
        subplot(211);
        plot(xx,Pout*1e-3.*Nl,'rx'); hold on;
        boxplot(kX,'Positions',xx,'Whisker',10); hold on;
        xticklabels(xtl);
        xlabel(['Penetration, \% (Feeder N',num2str(modeli),')']); 
        ylabel('Hosting capacity $\Phi$, kW'); grid on;
        xs = axis; axis([xs(1:2),0,xs(4)])
        lgnd = legend('$\Phi_{\mathrm{5\%}}$');
        set(lgnd,'Interpreter','Latex','FontSize',12,'Location','SouthEast');


        subplot(212);
        plot(xx,Pout*1e-3,'ro'); hold on;
        boxplot(X*1e-3,'Positions',xx,'Whisker',10); hold on;
        xticklabels(xtl)
        xlabel(['Penetration, \% (Feeder N',num2str(modeli),')']); 
        ylabel('Generator power $\phi$, kW'); grid on;
        if modeli==3
            xs=axis; axis([xs(1:2),0,25]);
        else
            xs = axis; axis([xs(1:2),0,40]);
        end
        lgnd = legend('$\phi_{\mathrm{5\%}}$');
        set(lgnd,'Interpreter','Latex','FontSize',12);

        FL = [FD,'variable_N_',num2str(modeli),'Pstr'];
        if exist('plotExport','var')
           export_fig(gcf,[FL,'.pdf'],'-pdf','-transparent'); close;
        end
    end

    if exist('thesisPlot','var')
        close
        PositionSet = [100 150 650 350];
        fig = figure('Position',PositionSet);
        subplot(121);
        boxplot(kX,'Positions',xx,'Whisker',10,'colors',colorOrder(1,:),'Whisker',1000); hold on;
        xticklabels(xtl);
        xlabel(['Fraction of loads with PV, \%']); 
        ylabel('Power hosting capacity curve $f_{\mathrm{Pwr}}$, kW'); grid on;

        xs = axis; axis([xs(1:2),0,Pout(end)*1e-3.*Nl(end)*1.5]); xs = axis;

        dP = xs(4)/PositionSet(4);
        MSize = 5;

        plot(xx,max(X).*Nl*1e-3 + dP*MSize,'v','color',colorOrder(1,:),'MarkerFaceColor','w','MarkerSize',MSize);
        plot(xx,min(X).*Nl*1e-3 - dP*MSize,'^','color',colorOrder(1,:),'MarkerFaceColor','w','MarkerSize',MSize);

        pp = plot(xx,Pout*1e-3.*Nl,'x','color',colorOrder(2,:));
        lgnd = legend([pp],'Total power, $P^{\mathrm{5\%}}_{\mathrm{PV}}$');
        set(lgnd,'Interpreter','Latex','FontSize',12);

        subplot(122);
        boxplot(X*1e-3,'Positions',xx,'Whisker',10,'colors',colorOrder(1,:),'Whisker',1000); hold on;

        xticklabels(xtl)
        xlabel(['Fraction of loads with PV, \%']); 
        ylabel('Power per generator, $f_{\mathrm{Pwr}}/N_{\mathrm{Gen}}$, kW'); grid on;
        xs = axis; axis([xs(1:2),0,Pout(1)*1e-3*1.25]); xs = axis;

        dP = xs(4)/PositionSet(4);
        MSize = 5;
        plot(xx,max(X)*1e-3 + dP*MSize,'v','color',colorOrder(1,:),'MarkerFaceColor','w','MarkerSize',MSize);
        plot(xx,min(X)*1e-3 - dP*MSize,'^','color',colorOrder(1,:),'MarkerFaceColor','w','MarkerSize',MSize);
        axis(xs);

        pp = plot(xx,Pout*1e-3,'o','color',colorOrder(2,:));
        lgnd = legend([pp],'Power per gen., $P^{\mathrm{5\%}}_{\mathrm{Gen}}$');
        set(lgnd,'Interpreter','Latex','FontSize',12);

        SD = ['C:\Users\',getenv('username'),'\Documents\DPhil\thesis\c4tech2\c4figures\'];
        FL = [SD,'monte_carlo_',model];
        if exist('plotExport','var')
            export_fig(gcf,FL);
            export_fig(gcf,[FL,'.pdf'],'-pdf','-transparent'); close;
        end
    end





    % figure;
    % boxplot(Vub,'Positions',Nl,'Whisker',10);
    % xticklabels(cellstr(num2str(Nl')))
    % title('Voltage unbalance');
    % xlabel('# houses'); ylabel('Voltage unbalance, |V_n_s|/|V_p_s| (%)'); grid on;


    % figure;
    % boxplot(kX(1:Ns/2,:),'Positions',Nl,'Whisker',10,'Width',Nl(end)/30); hold on;
    % boxplot(kX(Ns/2 + 1:end,:),'Positions',Nl+Nl(end)/25,'Whisker',10,'Width',Nl(end)/30);
    % xticklabels(cellstr(num2str(Nl')));
    % title('Total Power');
    % xlabel('Number of loads'); ylabel('kW'); grid on;

    % % subplot(122);
    % % xticklabels(cellstr(num2str(Nl')))
    % % title('Total Power');
    % % xlabel('Number of loads'); ylabel('kW'); grid on;

    % mn0 = min(kX);
    % mx0 = max(kX);
    % qnt = quantile(X,[0.05,0.25,0.75,0.95])
    % mdn0 = median(kX);

    %
    % mn0 = min(kX(1:Ns/2,:));
    % mx0 = max(kX(1:Ns/2,:));
    % qnt0 = quantile(kX(1:Ns/2,:),[0.05,0.25,0.75,0.95])
    % mdn0 = median(kX(1:Ns/2,:));

    % mn1 = min(kX(Ns/2 + 1:end,:));
    % mx1 = max(kX(Ns/2 + 1:end,:));
    % qnt1 = quantile(kX(Ns/2 + 1:end,:),[0.05,0.25,0.75,0.95])
    % mdn1 = median(kX(Ns/2 + 1:end,:));

    % figure;
    % plot(Nl,qnt0,'x'); hold on;
    % plot(Nl,qnt1,'o');

    % figure;
    % e_qnt = 100*(qnt0-qnt1)./qnt0;
    % plot(Nl,e_qnt,'x');
    % xlabel('# houses'); ylabel('% error'); grid on;
    % legend('5%','25%','75%','95%')
    % save(sn,'modeli','X','Nl','kX','mc_time','gen_pf','Vub')

end

if exist('thesisCasesTable','var')    
    display(caseStudies)
    
    headerCol = {};
    for ii = 1:numel(model_is)
        model = models(model_is(ii));
        headerCol{ii} = modelsTidy(model{:});
    end
    
    
    mat = caseStudies;
    headerRow = {'Feeder','$P_{\mathrm{Gen}\,100\%}$','$P_{\mathrm{Gen}}^{1\%}$',...
                          '$P_{\mathrm{Gen}}^{5\%}$','$P_{\mathrm{Gen}}^{10\%}$',...
                          '$P_{\mathrm{Gen}}^{5\%}$','$P_{\mathrm{Gen},\,\alpha_{\mathrm{PV}}=40\%}^{1\%}$'};
%         headerRow = {'Feeder','$P_{\mathrm{Gen}\,100\%}$','$P_{\mathrm{Gen}\,25\%}^{1\%}$',...
%                               '$P_{\mathrm{Gen}\,25\%}^{5\%}$','$P_{\mathrm{Gen}\,25\%}^{10\%}$',...
%                               '$P_{\mathrm{Gen}\,50:100\%}^{5\%}$','$P_{\mathrm{Gen}\,50\%}^{5\%}$',...
%                               '$f_{\mathrm{Pwr}}(P_{\mathrm{Gen}\,100\%})$, \%'};
    caption = 'DG Policy Decisions, based on stochastic hosting capacity results. All powers are in kW.';
    formatCol = {'$%.2f$','$%.2f$','$%.2f$','$%.2f$','$%.2f$','$%.2f$'};
    Tnm = 'thesisCasesTable';
    TF = ['C:\Users\',getenv('username'),'\Documents\DPhil\thesis\c4tech2\c4tables\',Tnm];
    if exist('tableExport','var')
        matrix2latex(mat,TF,'label',Tnm,'formatColumns',formatCol,'alignment','l','caption',caption,...
                    'headerRow',headerRow,'headerColumn',headerCol)
    end
end