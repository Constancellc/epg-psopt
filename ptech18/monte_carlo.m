% clear all; close all;
tic
WD = 'C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\ptech18';
% addpath('lin_functions');

modeli = 1; % CHOOSE model here
nl = linspace(0,1,7); % number of load here

models = {'eulv','n1f1','n2f1','n3f1','n4f1'};
fn{1} = [WD,'\LVTestCase_copy\master_z_g'];
fn{2} = [WD,'\manchester_models\network_1\Feeder_1\master_g'];
fn{3} = [WD,'\manchester_models\network_2\Feeder_1\master_g'];
fn{4} = [WD,'\manchester_models\network_3\Feeder_1\master_g'];
fn{5} = [WD,'\manchester_models\network_4\Feeder_1\master_g'];

[~,DSSObj,DSSText] = DSSStartup;
DSSCircuit = DSSObj.ActiveCircuit;
DSSSolution = DSSCircuit.Solution;
LDS = DSSCircuit.loads;
DSSText.command=['Compile (',fn{modeli},')'];
model = models{modeli};

load([WD,'\lin_models\',model]);

nl(1) = 1e-4;
Nl = ceil(nl*LDS.count);
Ns = 30; % number of samples

Vb = 230;
vp = 1.10;
vmax = Vb*vp;
p0 = 300; % watts
% gen_pf = -0.95;
gen_pf = 1.00;
aa = exp(1i*2*pi/3)

sn = [WD,'\lin_models\mc_out_',model]
% sn = [WD,'\lin_models\mc_out_',model,'_upf']

qgen = sin(acos(abs(gen_pf)))/gen_pf;

vmag=zeros(Ns,numel(Nl));
X=zeros(Ns,numel(Nl));
Vub=zeros(Ns,numel(Nl));
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
for i = 1:numel(Nl)
	xhp0 = xhy0(1:numel(xhy0)/2);
	fxp0 = find(xhp0);
	
	for j = 1:Ns
		rdi = randperm(LDS.count,Nl(i));
		xhs = sparse(zeros(size(xhp0)));

		xhs(fxp0(rdi)) = 1;
		xhs = [xhs;xhs*qgen];
		Mk = My*xhs;
		A = real(Mk./ang0);
	
		[X(j,i),FVal,exitflag] = linprog(f,A,b);
		
		Vout = My*(xhy0 + xhs*X(j,i)) + a;
		Vps = PSm*Vout;
		Vns = NSm*Vout;
		Vub(j,i) = max(100*abs(Vns)./abs(Vps)); % in %
	end
	% now compile and compare to opendss version

	% DSSText.command=['Batchedit load..* vmaxpu=10 vminpu=0.1'];
	% DSSText.command=['Batchedit generator..* vmaxpu=10 vminpu=0.1'];
	% DSSText.command=['Batchedit generator..* pf=',num2str(gen_pf)];
	% DSSText.command=['set loadmult=',num2str(p0/1e3)];
	% DSSText.command=['set genmult=',num2str(X(i)/1e3)];
	% DSSSolution.Solve
	% vmag(j,i) = max(DSSCircuit.AllBusVmag(4:end))/Vb;
end
mc_time=toc;
kX = X.*Nl*1e-3;

figure;
subplot(121);
boxplot(X*1e-3,'Positions',Nl);
xticklabels(cellstr(num2str(Nl')))
title('Power per House');
xlabel('# houses'); ylabel('kW/house'); grid on;

subplot(122);
boxplot(kX,'Positions',Nl);
xticklabels(cellstr(num2str(Nl')))
title('Total Power');
xlabel('Number of loads'); ylabel('kW'); grid on;

figure;
boxplot(Vub,'Positions',Nl);
xticklabels(cellstr(num2str(Nl')))
title('Voltage unbalance');
xlabel('# houses'); ylabel('Voltage unbalance, |V_n_s|/|V_p_s|'); grid on;

save(sn,'modeli','X','Nl','kX','mc_time','gen_pf','Vub')