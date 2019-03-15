% clear all; close all;

set(0,'DefaultTextInterpreter','latex')
% WD = 'C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\ptech18';
WD = pwd;
% FD = 'C:\Users\Matt\Documents\DPhil\pesgm19\pesgm19_paper\figures\';
FD = 'C:\Users\Matt\Documents\DPhil\papers\pesgm19\pesgm19_poster\figures\'; % poster version
addpath('lin_functions');
models = {'eulv','n1f1','n2f1','n3f1','n4f1'};
model = models{1};
fn{1} = [WD,'\LVTestCase_copy\master_z_g'];
fn{2} = [WD,'\manchester_models\network_1\Feeder_1\master_g'];
fn{3} = [WD,'\manchester_models\network_2\Feeder_1\master_g'];
fn{4} = [WD,'\manchester_models\network_3\Feeder_1\master_g'];
fn{5} = [WD,'\manchester_models\network_4\Feeder_1\master_g'];

Vb = 230;
vp = 1.10;
vmax = Vb*vp;
p0 = 300; % watts
gen_pf = -0.95;

qgen = sin(acos(abs(gen_pf)))/gen_pf;

[~,DSSObj,DSSText] = DSSStartup;
DSSCircuit = DSSObj.ActiveCircuit;
DSSSolution = DSSCircuit.Solution;

vmag=zeros(size(fn));
X=zeros(size(fn));
tic
for i = 1:numel(models)
% for i = 1:1
	model = models{i};
	load([WD,'\lin_models\',model]);

	Vh0 = My*xhy0 + a;
	
	xhp0 = xhy0(1:numel(xhy0)/2);
	Mk = My*([xhp0~=0;zeros(size(xhp0))]);
	Mk = My*([xhp0~=0;(xhp0~=0)*qgen]);

	ang0 = diag(exp(1i*angle(Vh0)));

	Va0 = real(ang0\Vh0);
	Mk0 = real(ang0\Mk);

	Vmax = vmax*ones(size(Mk0));

	A = Mk0;
	b = Vmax - Va0;
	f = -1;

	[X(i),FVal,exitflag] = linprog(f,A,b);
	
	% now compile and compare to opendss version
	DSSText.Command=['Compile (',fn{i},')'];
	DSSText.Command=['Batchedit load..* vmaxpu=10 vminpu=0.1'];
	DSSText.Command=['Batchedit generator..* vmaxpu=10 vminpu=0.1'];
	DSSText.Command=['Batchedit generator..* pf=',num2str(gen_pf)];
	DSSText.Command=['set loadmult=',num2str(p0/1e3)];
	DSSText.Command=['set genmult=',num2str(X(i)/1e3)];
	DSSSolution.Solve
	vmag(i) = max(DSSCircuit.AllBusVmag(4:end))/Vb;
end
toc
display(vmag);
display(X);

Vt = [1.05, 1,1,1,1]*240/230;

dV =  [ones(1,5)*vp;vmag] - ones(2,1)*Vt;

FN = [FD,'lin_program_vldtPstr'];
%% PAPER VERSION
fig = figure('Color','White','Position',[100 150 550 250]);
subplot(121);
bar(X/1e3)
xlabel('Feeder');
ylabel('Power per house, kW (n = 100\%)')
axis([-inf inf 0 11]);
xticklabels({'EU LV','N1.1','N2.1','N3.1','N4.1'})
subplot(122);
bar(dV');
xlabel('Feeder');
ylabel('$v^{\mathrm{max}} - v^{\mathrm{Sub}}$ (pu)');
axis([-inf inf 0 0.09]);
xticklabels({'EU LV','N1.1','N2.1','N3.1','N4.1'})
legend('Predicted (Lin. Model)','Actual (OpenDSS)','Location','NorthWest');

% export_fig(fig,FN);
% export_fig(fig,[FN,'.pdf'],'-dpdf');
% saveas(fig,FN,'meta');
%% POSTER VERSION
fig = figure('Color','White','Position',[100 150 260 400]);
subplot(211);
bar(X/1e3)
xlabel('Feeder ID');
ylabel('Power per house, kW')
axis([-inf inf 0 11]);
xticklabels({'N1','N2','N3','N4','N5'})
subplot(212);
bar(dV');
xlabel('Feeder ID');
ylabel('Max voltage rise ($\Delta v^{\mathrm{max}}$), pu');
axis([-inf inf 0 0.12]);
xticklabels({'N1','N2','N3','N4','N5'})
legend('Linear Model','OpenDSS','Location','NorthWest');

FN = [FD,'lin_program_vldtPstr'];
export_fig(fig,[FN,'.pdf'],'-dpdf','-transparent'); close



















