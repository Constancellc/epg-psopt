clear all; % close all;
tic
set(0,'DefaultTextInterpreter','Latex')

posterPlt = 1;

% FD = 'C:\Users\Matt\Documents\DPhil\pesgm19\pesgm19_paper\figures\'; % paper version
FD = 'C:\Users\Matt\Documents\DPhil\papers\pesgm19\pesgm19_poster\figures\'; % poster version

modeli = 3; % CHOOSE model here
nl = linspace(0,1,7); % number of load here
nl = linspace(0,1,13);
nl(1) = 1e-4;
% nl = 0.1;
% nl = linspace(0.4,0.6,5); % number of load here
mode = 'Vfix'; % fix voltages
% mode = 'Wfix'; % fix powers (W)
rng(0);
frac_set = 0.05;
% frac_set = 0.95;

models = {'eulv','n1f1','n2f1','n3f1','n4f1'};
fn{1} = [pwd,'\LVTestCase_copy\master_z_g'];
fn{2} = [pwd,'\manchester_models\network_1\Feeder_1\master_g'];
fn{3} = [pwd,'\manchester_models\network_2\Feeder_1\master_g'];
fn{4} = [pwd,'\manchester_models\network_3\Feeder_1\master_g'];
fn{5} = [pwd,'\manchester_models\network_4\Feeder_1\master_g'];

[~,DSSObj,DSSText] = DSSStartup;
DSSCircuit = DSSObj.ActiveCircuit;
DSSSolution = DSSCircuit.Solution;
LDS = DSSCircuit.loads;
DSSText.command=['Compile (',fn{modeli},')'];
model = models{modeli};

load([pwd,'\lin_models\',model]);

Nl = ceil(nl*LDS.count);

Ns = 1000; % number of samples
% Ns = 100; % number of samples

Vb = 230;
vp = 1.10;
vmax = Vb*vp;
% gen_pf = -0.95;
% gen_pf = 1.00;
gen_pf = 0.95;
aa = exp(1i*2*pi/3);

sn = [pwd,'\lin_models\mc_out_',model];
% sn = [pwd,'\lin_models\mc_out_',model,'_upf']

qgen = sin(acos(abs(gen_pf)))/gen_pf;

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

PSm = sparse(kron(eye(numel(xhy0)/6),[1,aa,aa^2]))/(sqrt(3)*(Vb*sqrt(3)));
NSm = sparse(kron(eye(numel(xhy0)/6),[1,aa^2,aa]))/(sqrt(3)*(Vb*sqrt(3)));

Vmax = vmax*ones(size(Vh0));
b = Vmax - Va0;

xhp0 = xhy0(1:numel(xhy0)/2);
fxp0 = find(xhp0);

PPa = [];
PPb = [];

if strcmp(mode,'Wfix')
    xhs = sparse(zeros(size(xhp0)));
    xhs(fxp0) = 1;
    xhs = [xhs;xhs*qgen];
    Mk = My*xhs;
    A = real(Mk./ang0);
    Anew = A(xhp0~=0);
    bnew = b(xhp0~=0);
    [P100,FVal,exitflag] = linprog(f,Anew,bnew);
    
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
            Vps = PSm*Vout;
            Vns = NSm*Vout;
            Vub(j,i) = max(100*abs(Vns)./abs(Vps)); % in %
            
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
    Pout = mean([Pa,Pb])*LDS.count/mean(Nl);
    display(count);
    display(Pout);
end

tic
if strcmp(mode,'Vfix')
    for i = 1:numel(Nl)
        toc
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
            Vps = PSm*Vout;
            Vns = NSm*Vout;
            Vub(j,i) = max(100*abs(Vns)./abs(Vps)); % in %
        end
        Pout(i) = quantile(X(:,i),frac_set);
    end
    kX = X.*Nl*1e-3;
end
% mc_time=toc;
% display(mc_time);

fig = figure('Color','White','Position',[100 150 550 270]);

FL = [FD,'variable_N_',num2str(modeli)];

if modeli==3
    fdr = 'N2.1';
    xx = 100*Nl./Nl(end);
    xtl = cellstr(num2str(xx','%.0f'));
    xtl(2:3:end) = {''};
    xtl(3:3:end) = {''};
    xtl(1) = {'0.57'};
elseif modeli==5
    fdr = 'N4.1';
    xx = 100*Nl./Nl(end);
    xtl = cellstr(num2str(xx','%.0f'));
    xtl(2:3:end) = {''};
    xtl(3:3:end) = {''};
    xtl(1) = {'4.2'};
else
    fdr = num2str(modeli);
    xtl = cellstr(num2str(Nl'));
end

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
else
    xs = axis; axis([xs(1:2),0,40]);
end
lgnd = legend('$\phi_{\mathrm{5\%}}$');
set(lgnd,'Interpreter','Latex','FontSize',12);

% export_fig(gcf,FL);
% export_fig(gcf,[FL,'.pdf'],'-pdf');
% saveas(gcf,FL,'meta')


if posterPlt
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
    export_fig(gcf,[FL,'.pdf'],'-pdf','-transparent'); close;
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






