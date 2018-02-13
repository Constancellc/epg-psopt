% unbalanced_cpf is use to see how the error changes as a function of
% imbalance on the system.
%
% We look at two types of unbalance: an 'a' type unbalance: one phase lo and
% the other two identical/hi; 'b' type, in a 1-2-3 linear increase (e.g.
% 0.7/1.0/1.3 for 1kw).


close all; clear all; clc;

WD = 'C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\risk_day';
cd(WD); addpath('mtd_fncs');

fg_lc = 'C:\Users\chri3793\Documents\DPhil\risk_day\rd_ppt\figures\';   
fg_ps = [200 300 400 400];

set(0,'defaulttextinterpreter','latex'); set(0,'defaultaxesfontsize',14);
set(0,'defaulttextfontsize',14);

S0 = 1 + 1i*sqrt( (1 - 0.95^2)/(0.95^2) );
%%
% load('lvtestcase_lin.mat');

G = [1 -1 0;0 1 -1; -1 0 1]; %gamma matrix
fn = [WD,'\LVTestCase_copy\Master_z'];
% fn = [WD,'\LVTestCase_copy\Master'];

GG.filename = fn; 
GG.filename_v = [fn,'_v']; 
GG.filename_y = [fn,'_y'];
%%
% Run the nominal DSS
[~, DSSObj, DSSText] = DSSStartup;
DSSCircuit=DSSObj.ActiveCircuit;
DSSSolution=DSSCircuit.Solution;
DSSText.command=['Compile (',GG.filename,')'];

YZNodeOrder = DSSCircuit.YNodeOrder;
%% LOAD the linear models (from lin_cpf.m):
DD = load([WD,'\datasets\lvtestcase_lin1']);
a1 = DD.a1; My1 = DD.My1;
DD = load([WD,'\datasets\lvtestcase_lin2']);
a2 = DD.a2; My2 = DD.My2;
DD = load([WD,'\datasets\lvtestcase_lin3']);
a3 = DD.a3; My3 = DD.My3;
DD = load([WD,'\datasets\lvtestcase_lin4']);
a4 = DD.a4; My4 = DD.My4;

sY = DD.sY; Ybus = DD.Ybus; v0 = DD.v0;

%%
k = (-0.75:0.005:1.75);
% k = (-0.75:0.05:1.75);
Ybus_sp = sparse(Ybus);

cpf = zeros(4,numel(k));
cpfl = zeros(4,numel(k));

vc = zeros(numel(sY) - 3,4);
Sls = zeros(4,numel(k));

sy = zeros(size(k));
sl = zeros(size(k));
vn = zeros(size(k));
tic % 16 seconds (numel(k) = 250)
for i = 1:numel(k)
    DSSCircuit = set_loads(DSSCircuit,S0*k(i));
    DSSSolution.Solve;

    YNodeVarray = DSSCircuit.YNodeVarray';
    YNodeV = YNodeVarray(1:2:end) + 1i*YNodeVarray(2:2:end);
    DSSls = DSSCircuit.Losses;
    sl(i) = (DSSls(1) + 1i*DSSls(2))/1e3;
    
    [B,V,I,S,D] = ld_vals( DSSCircuit );
    [~,~,~,sY] = calc_sYsD( YZNodeOrder,B,I,S,D );
    
    sy(i) = sY(143); % for plotting the actual load
    
    xhy = -1e3*[real(sY(4:end));imag(sY(4:end))];

    [ vc(:,1),~,~,Sls(1,i) ] = lin_pf_y( My1,a1,Ybus_sp,v0,xhy );
    [ vc(:,2),~,~,Sls(2,i) ] = lin_pf_y( My2,a2,Ybus_sp,v0,xhy );
    [ vc(:,3),~,~,Sls(3,i) ] = lin_pf_y( My3,a3,Ybus_sp,v0,xhy );
    [ vc(:,4),~,~,Sls(4,i) ] = lin_pf_y( My4,a4,Ybus_sp,v0,xhy );

    for j = 1:size(vc,2)
        cpf(j,i) = norm(vc(:,j) - YNodeV(4:end))/norm(YNodeV(4:end));
        cpfl(j,i) = abs(Sls(j,i) - sl(i))/abs(sl(i));
    end
    vn(i) = norm(YNodeV(4:end));
end
toc

%%
fg_nm = [fg_lc,'lin_cpf_sngl'];
fig = figure('Color','White','Position',fg_ps);

plot(k,cpf(2,:));
xlabel('Load $k\,,$ kW'); ylabel('Error, $||\tilde{v} - v||_{2}/||v||_{2}$');
xlim([-inf inf]);

lgnd=legend('$\hat{S} = 0.6$ kW');
set(lgnd,'Interpreter','Latex')

% export_fig(fig,fg_nm);
% export_fig(fig,[fg_nm,'.pdf'],'-dpdf');

%%
fg_nm = [fg_lc,'lin_cpf'];
fig = figure('Color','White','Position',fg_ps);

plot(k,cpf);
xlabel('Load $k\,,$ kW'); ylabel('Error, $||\tilde{v} - v||_{2}/||v||_{2}$');
lgnd=legend('$\hat{S} = 0.2$ kW','$\hat{S} = 0.6$ kW','$\hat{S} = 1.0$ kW','$\hat{S} = 1.4$ kW');%
set(lgnd,'Interpreter','Latex','Location','NorthWest')
xlim([-inf inf]);

% export_fig(fig,fg_nm);
% export_fig(fig,[fg_nm,'.pdf'],'-dpdf');

%%
fg_nm = [fg_lc,'lin_cpfl'];
fig = figure('Color','White','Position',fg_ps);

plot(k,100*cpfl);
xlabel('Load $k\,,$ kW'); ylabel('Error, $|\tilde{S}_{l} - S_{l}|/|S_{l}|\,, \%$');
lgnd=legend('$\hat{S} = 0.2$ kW','$\hat{S} = 0.6$ kW','$\hat{S} = 1.0$ kW','$\hat{S} = 1.4$ kW');%
set(lgnd,'Interpreter','Latex')
xlim([-inf inf]);

% export_fig(fig,fg_nm);
% export_fig(fig,[fg_nm,'.pdf'],'-dpdf');
