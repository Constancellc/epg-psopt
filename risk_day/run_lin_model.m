clear all; close all; clc;

WD = 'C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\risk_day';
cd(WD); addpath('mtd_fncs');

fg_lc = 'C:\Users\chri3793\Documents\DPhil\risk_day\rd_ppt\figures\';   
fg_ps = [200 300 400 400]; fg_ps_sht = [200 300 400 240];

set(0,'defaulttextinterpreter','latex'); set(0,'defaultaxesfontsize',14);
set(0,'defaulttextfontsize',14);

S0 = 1 + 1i*sqrt( (1 - 0.95^2)/(0.95^2) );
G = [1 -1 0;0 1 -1; -1 0 1]; %gamma matrix
%% Run the nominal feeder to 
fn = [WD,'\LVTestCase_copy\Master_z'];
GG.filename = fn; GG.filename_v = [fn,'_v']; GG.filename_y = [fn,'_y'];

% Run the nominal DSS
[~, DSSObj, DSSText] = DSSStartup;
DSSCircuit=DSSObj.ActiveCircuit;
DSSSolution=DSSCircuit.Solution;
DSSText.command=['Compile (',GG.filename,')'];
YZNodeOrder = DSSCircuit.YNodeOrder;
%% Load linear models
DD = load([WD,'\datasets\lvtestcase_lin1']);
a1 = DD.a1; My1 = DD.My1;
DD = load([WD,'\datasets\lvtestcase_lin2']);
a2 = DD.a2; My2 = DD.My2;
DD = load([WD,'\datasets\lvtestcase_lin3']);
a3 = DD.a3; My3 = DD.My3;
DD = load([WD,'\datasets\lvtestcase_lin4']);
a4 = DD.a4; My4 = DD.My4;

sY = DD.sY; Ybus = DD.Ybus; v0 = DD.v0;
Ybus_sp = sparse(Ybus);
%%
n = 10000;
GC = load([WD,'\datasets\gamma_consts.mat']);
GM = gamrnd(GC.a,1/GC.b,[n,55]);

sl = zeros(1,n); st = zeros(1,n); Ssm = zeros(1,n);
Sfd = zeros(4,n); Sls = zeros(4,n); Sld = zeros(4,n);

tic % n=10000: ~200 seconds
for i = 1:n
    DSSCircuit = set_load_prfl(DSSCircuit,S0*GM(i,:));
    DSSSolution.Solve;
    YNodeVarray = DSSCircuit.YNodeVarray';
    YNodeV = YNodeVarray(1:2:end) + 1i*YNodeVarray(2:2:end);
    DSSls = DSSCircuit.Losses;
    sl(i) = (DSSls(1) + 1i*DSSls(2))/1e3;
    DSSst = DSSCircuit.TotalPower;
    st(i) = (DSSst(1) + 1i*DSSst(2));
    
    [B,V,I,S,D] = ld_vals( DSSCircuit );
    [~,~,~,sY] = calc_sYsD( YZNodeOrder,B,I,S,D );
    
    Ssm(i) = sum(sY);
        
    xhy = -1e3*[real(sY(4:end));imag(sY(4:end))];
    xhy = sparse(xhy);
    
    [ ~,Sfd(1,i),Sld(1,i),Sls(1,i) ] = lin_pf_y( My1,a1,Ybus_sp,v0,xhy );
    [ ~,Sfd(2,i),Sld(2,i),Sls(2,i) ] = lin_pf_y( My2,a2,Ybus_sp,v0,xhy );
    [ ~,Sfd(3,i),Sld(3,i),Sls(3,i) ] = lin_pf_y( My3,a3,Ybus_sp,v0,xhy );
    [ ~,Sfd(4,i),Sld(4,i),Sls(4,i) ] = lin_pf_y( My4,a4,Ybus_sp,v0,xhy );
end
toc

DLd = Sld + Ssm;
ST = Sfd+DLd;



%% ------------------------------ LOSS RESULTS
fig = figure('Color','White','Position',fg_ps);
fg_nm = [fg_lc,'Sl'];

n=12;
plot(sl(1:n),'ko'); hold on;
plot(Sls(:,1:n).','x');
axis equal;
xlabel('$P_{l}$'); ylabel('$Q_{l}$'); grid on;
lgnd = legend('True','$0.2\,\hat{S}$','$0.6\,\hat{S}$','$1.0\,\hat{S}$','$1.4\,\hat{S}$');
set(lgnd,'Location','NorthWest','Interpreter','Latex');

% export_fig(fig,fg_nm);
% export_fig(fig,[fg_nm,'.pdf'],'-dpdf');

%%
fig = figure('Color','White','Position',fg_ps_sht);
fg_nm = [fg_lc,'DSl'];

plot(sl(1:n).'- Sls(:,1:n).','x'); hold on;
% plot(Sls_sm - Sls_CC,'x'); hold on;

xlabel('$\Delta P_{l}$'); ylabel('$\Delta Q_{l}$'); grid on; axis equal;
lgnd = legend('$0.2\,\hat{S}$','$0.6\,\hat{S}$','$1.0\,\hat{S}$','$1.4\,\hat{S}$');
ylim(1e-3*[-5 8]);
set(lgnd,'Location','NorthWest','Interpreter','Latex');

% export_fig(fig,fg_nm);
% export_fig(fig,[fg_nm,'.pdf'],'-dpdf');
%%
fig = figure('Color','White','Position',fg_ps_sht);
fg_nm = [fg_lc,'DSl_hist'];

EPS = 100*abs(sl-Sls)./abs(sl);

histogram(EPS(1,:),'Normalization','pdf'); hold on;
histogram(EPS(2,:),'Normalization','pdf');
histogram(EPS(3,:),'Normalization','pdf');
histogram(EPS(4,:),'Normalization','pdf');
xlabel('$\epsilon = |\tilde{S}_{l}-S_{l}|/|S_{l}|\,,\, \%$');
ylabel('$p(\epsilon)$');

lgnd = legend('$0.2\,\hat{S}$','$0.6\,\hat{S}$','$1.0\,\hat{S}$','$1.4\,\hat{S}$');
set(lgnd,'Interpreter','Latex');
% export_fig(fig,fg_nm,'-r300');

%% --------------------------- SEPERATE into P/Q
EPS_p = 100*abs(real(sl-Sls))./abs(real(sl));

histogram(EPS_p(1,:),'Normalization','pdf'); hold on;
histogram(EPS_p(2,:),'Normalization','pdf');
histogram(EPS_p(3,:),'Normalization','pdf');
histogram(EPS_p(4,:),'Normalization','pdf');
xlabel('$\epsilon = |\tilde{S}_{l}-S_{l}|/|S_{l}|\,,\, \%$');
ylabel('$p(\epsilon)$');

lgnd = legend('$0.2\,\hat{S}$','$0.6\,\hat{S}$','$1.0\,\hat{S}$','$1.4\,\hat{S}$');
set(lgnd,'Location','NorthWest','Interpreter','Latex');
%%
EPS_q = 100*abs(imag(sl-Sls))./abs(imag(sl));

histogram(EPS_q(1,:),'Normalization','pdf'); hold on;
histogram(EPS_q(2,:),'Normalization','pdf');
histogram(EPS_q(3,:),'Normalization','pdf');
histogram(EPS_q(4,:),'Normalization','pdf');
xlabel('$\epsilon = |\tilde{S}_{l}-S_{l}|/|S_{l}|\,,\, \%$');
ylabel('$p(\epsilon)$');

lgnd = legend('$0.2\,\hat{S}$','$0.6\,\hat{S}$','$1.0\,\hat{S}$','$1.4\,\hat{S}$');
set(lgnd,'Location','NorthWest','Interpreter','Latex');




%% ------------------------ TOTAL POWER RESULTS
fig = figure('Color','White','Position',fg_ps);
fg_nm = [fg_lc,'ST'];

n = 30;
plot(-st(1:n),'ko'); hold on;
plot(ST(2,1:n),'rx');
plot(Ssm(1:n),'b+');
xlabel('$P_{T}$'); ylabel('$Q_{T}$'); axis equal; grid on;

lgnd = legend('True','Lin PF','Sum');
set(lgnd,'Interpreter','Latex','Location','NorthWest');

% export_fig(fig,fg_nm);
% export_fig(fig,[fg_nm,'.pdf'],'-dpdf');
%%
fig = figure('Color','White','Position',fg_ps_sht);
fg_nm = [fg_lc,'DST'];

plot(st(1:n)+ST(2,1:n),'rx'); hold on;
plot(st(1:n)+Ssm(1:n),'bx');
grid on; axis equal; 
xlabel('$\Delta P_{T}$'); ylabel('$\Delta Q_{T}$');

lgnd = legend('Lin PF','Sum');
set(lgnd,'Interpreter','Latex','Location','SouthEast');

% export_fig(fig,fg_nm);
% export_fig(fig,[fg_nm,'.pdf'],'-dpdf');
%%
fig = figure('Color','White','Position',fg_ps_sht);
fg_nm = [fg_lc,'DST_hist'];

histogram(log10(abs(st+Ssm)./abs(st)),'Normalization','pdf','EdgeColor','b','FaceColor','b'); hold on;
histogram(log10(abs(st+ST(2,:))./abs(st)),'Normalization','pdf','EdgeColor','r','FaceColor','r');
% xlim([0,1.35]);
xlabel('$\epsilon = \log_{10}(\,|\tilde{S}_{T}-S_{T}|/|S_{T}|\,) $');
ylabel('$p(\epsilon)$');

lgnd = legend('Sum','Lin PF');
set(lgnd,'Interpreter','Latex','Location','NorthWest');
% export_fig(fig,fg_nm,'-r300');



%% SUMMARY STATISTICS
clc
mean(abs(st))
mean(abs(ST(2,:)))
mean(abs(Ssm))

var(abs(st))
var(abs(ST(2,:)))
var(abs(Ssm))

mean(abs(sl))
mean(abs(Sls(2,:)))











