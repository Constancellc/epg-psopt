% clear all; close all;

WD = 'C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\ptech18';
% addpath('lin_functions');
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
% for i = 1:numel(models)
for i = 1:1
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
	DSSText.command=['Compile (',fn{i},')'];
	DSSText.command=['Batchedit load..* vmaxpu=10 vminpu=0.1'];
	DSSText.command=['Batchedit generator..* vmaxpu=10 vminpu=0.1'];
	DSSText.command=['Batchedit generator..* pf=',num2str(gen_pf)];
	DSSText.command=['set loadmult=',num2str(p0/1e3)];
	DSSText.command=['set genmult=',num2str(X(i)/1e3)];
	DSSSolution.Solve
	vmag(i) = max(DSSCircuit.AllBusVmag(4:end))/Vb;
end
toc
display(vmag);
display(X);



