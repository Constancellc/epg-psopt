clear all; close all; clc;

% WD = 'C:\Users\Matt\Documents\MATLAB\epg-psopt\risk_day';
WD = 'C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\risk_day';
cd(WD);

% fg_lc = 'C:\Users\Matt\Documents\DPhil\risk_day\rd_ppt\figures\';
fg_lc = 'C:\Users\chri3793\Documents\DPhil\risk_day\rd_ppt\figures\';   
fg_ps = [200 300 400 400];
fg_ps_sht = [200 300 400 240];

set(0,'defaulttextinterpreter','latex');
set(0,'defaultaxesfontsize',14);
set(0,'defaulttextfontsize',14);

%%
% CC = csvread('datasets/riskDayLoadsInAndOut.csv',1,0);
CC = csvread('datasets/riskDayLoadsInAndOut_z.csv',1,0);
%%
Sfd_CC = CC(:,1) + 1i*CC(:,2);
Sld = CC(:,3:2:end) + 1i*CC(:,4:2:end);
Sls_CC = Sfd_CC - sum(Sld,2);
Ssm = sum(Sld,2);

%%
% DD = load('datasets/lvtestcase_lin');

DD = load([WD,'\datasets\lvtestcase_lin1']);
a1 = DD.a1; My1 = DD.My1;
DD = load([WD,'\datasets\lvtestcase_lin2']);
a2 = DD.a2; My2 = DD.My2;
DD = load([WD,'\datasets\lvtestcase_lin3']);
a3 = DD.a3; My3 = DD.My3;
DD = load([WD,'\datasets\lvtestcase_lin4']);
a4 = DD.a4; My4 = DD.My4;

sY = DD.sY; Ybus = DD.Ybus; v0 = DD.v0;

sY0 = sY(find(sY));
sYidx = find(sY);
pf = mean(real(sY0)./abs(sY0));
qK = sqrt(1-pf^2)/pf;
Ybus_SP = sparse(Ybus);
%
Sfd_DD = zeros(size(Sld,2),4);
Sls_DD = zeros(size(Sld,2),4);
Sld_DD = zeros(size(Sld,2),4);
% SLD_DD = zeros(size(Sld,2),4);
tic % ~11 seconds x4 = ~45 seconds
for i = 1:size(Sld,1)

    sY(sYidx)=Sld(i,:);
    xhy = sparse( -1e3*[real(sY(4:end));imag(sY(4:end))] );

    
    [ ~,Sfd_DD(i,1),Sld_DD(i,1),Sls_DD(i,1) ] = lin_pf_y( My1,a1,Ybus_SP,v0,xhy );
    [ ~,Sfd_DD(i,2),Sld_DD(i,2),Sls_DD(i,2) ] = lin_pf_y( My2,a2,Ybus_SP,v0,xhy );
    [ ~,Sfd_DD(i,3),Sld_DD(i,3),Sls_DD(i,3) ] = lin_pf_y( My3,a3,Ybus_SP,v0,xhy );
    [ ~,Sfd_DD(i,4),Sld_DD(i,4),Sls_DD(i,4) ] = lin_pf_y( My4,a4,Ybus_SP,v0,xhy );

%     
%     vc = My1*xhy + a1;
%     Vlin = [v0;vc];
%     Slin = Vlin.*conj(Ybus_SP*Vlin)/1e3;
%     Sfd_DD(i,1) = sum(Slin(1:3));
%     Sld_DD(i,1) = sum(Slin(sYidx));
%     SLD_DD(i,1) = sum(Slin(4:end));
% %     Sls_DD(i,1) = sum(Slin);
%     
% 
%     vc = My2*xhy + a2;
%     Vlin = [v0;vc];
%     Slin = Vlin.*conj(Ybus_SP*Vlin)/1e3;
%     Sfd_DD(i,2) = sum(Slin(1:3));
%     Sld_DD(i,2) = sum(Slin(sYidx));
%     SLD_DD(i,2) = sum(Slin(4:end));
% %     Sls_DD(i,2) = sum(Slin);
%     
%     vc = My3*xhy + a3;
%     Vlin = [v0;vc];
%     Slin = Vlin.*conj(Ybus_SP*Vlin)/1e3;
%     Sfd_DD(i,3) = sum(Slin(1:3));
%     Sld_DD(i,3) = sum(Slin(sYidx));
%     SLD_DD(i,3) = sum(Slin(4:end));
% %     Sls_DD(i,3) = sum(Slin);
%     
%     vc = My4*xhy + a4;
%     Vlin = [v0;vc];
%     Slin = Vlin.*conj(Ybus_SP*Vlin)/1e3;
%     Sfd_DD(i,4) = sum(Slin(1:3));
%     Sld_DD(i,4) = sum(Slin(sYidx));
%     SLD_DD(i,4) = sum(Slin(4:end));
% %     Sls_DD(i,4) = sum(Slin);
end
toc

% DLd = Sld_DD + Ssm;
% Sfd_DD = Sfd_DD - DLd;
% Sls_sm = zeros(size(Ssm));
% Sls_DD = Sfd_DD + Sld_DD;

% DLD = SLD_DD - Sld_DD;
% plot(DLd,'x');
% plot(DLD,'x');

%% ------------------------------ LOSS RESULTS
fig = figure('Color','White','Position',fg_ps);
fg_nm = [fg_lc,'Sl'];

n=30;
plot(Sls_CC(1:n),'ko'); hold on;
plot(Sls_DD(1:n,:),'x');
% plot(Sls_sm(1:n,:),Sls_sm(1:n,:),'kx');
axis equal;
xlabel('$P_{l}$'); ylabel('$Q_{l}$'); grid on;
lgnd = legend('True','$0.2\,\hat{S}$','$0.6\,\hat{S}$','$1.0\,\hat{S}$','$1.4\,\hat{S}$');
set(lgnd,'Location','NorthWest','Interpreter','Latex');

% export_fig(fig,fg_nm);
% export_fig(fig,[fg_nm,'.pdf'],'-dpdf');

%%
fig = figure('Color','White','Position',fg_ps_sht);
fg_nm = [fg_lc,'DSl'];

plot(Sls_DD - Sls_CC,'x'); hold on;
% plot(Sls_sm - Sls_CC,'x'); hold on;

xlabel('$\Delta P_{l}$'); ylabel('$\Delta Q_{l}$'); grid on; axis equal;
lgnd = legend('$0.2\,\hat{S}$','$0.6\,\hat{S}$','$1.0\,\hat{S}$','$1.4\,\hat{S}$');
set(lgnd,'Location','NorthWest','Interpreter','Latex');

% export_fig(fig,fg_nm);
% export_fig(fig,[fg_nm,'.pdf'],'-dpdf');

%%
fig = figure('Color','White','Position',fg_ps_sht);
fg_nm = [fg_lc,'DSl_hist'];

EPS = 100*abs(Sls_CC-Sls_DD)./abs(Sls_CC);

histogram(EPS(:,1),'Normalization','pdf'); hold on;
histogram(EPS(:,2),'Normalization','pdf');
histogram(EPS(:,3),'Normalization','pdf');
histogram(EPS(:,4),'Normalization','pdf');
xlabel('$\epsilon = |\tilde{S}_{l}-S_{l}|/|\tilde{S}_{l}|\,,\, \%$');
ylabel('$p(\epsilon)$');

lgnd = legend('$0.2\,\hat{S}$','$0.6\,\hat{S}$','$1.0\,\hat{S}$','$1.4\,\hat{S}$');
set(lgnd,'Location','NorthWest','Interpreter','Latex');
export_fig(fig,fg_nm,'-r300');
%% --------------------------- SEPERATE into P/Q
EPS_p = 100*abs(real(Sls_CC-Sls_DD))./abs(real(Sls_CC));

histogram(EPS_p(:,1),'Normalization','pdf'); hold on;
histogram(EPS_p(:,2),'Normalization','pdf');
histogram(EPS_p(:,3),'Normalization','pdf');
histogram(EPS_p(:,4),'Normalization','pdf');
xlabel('$\epsilon = |\tilde{S}_{l}-S_{l}|/|\tilde{S}_{l}|\,,\, \%$');
ylabel('$p(\epsilon)$');

lgnd = legend('$0.2\,\hat{S}$','$0.6\,\hat{S}$','$1.0\,\hat{S}$','$1.4\,\hat{S}$');
set(lgnd,'Location','NorthWest','Interpreter','Latex');
%%
EPS_q = 100*abs(imag(Sls_CC-Sls_DD))./abs(imag(Sls_CC));

histogram(EPS_q(:,1),'Normalization','pdf'); hold on;
histogram(EPS_q(:,2),'Normalization','pdf');
histogram(EPS_q(:,3),'Normalization','pdf');
histogram(EPS_q(:,4),'Normalization','pdf');
xlabel('$\epsilon = |\tilde{S}_{l}-S_{l}|/|\tilde{S}_{l}|\,,\, \%$');
ylabel('$p(\epsilon)$');

lgnd = legend('$0.2\,\hat{S}$','$0.6\,\hat{S}$','$1.0\,\hat{S}$','$1.4\,\hat{S}$');
set(lgnd,'Location','NorthWest','Interpreter','Latex');
%% ------------------------ TOTAL POWER RESULTS
fig = figure('Color','White','Position',fg_ps);
fg_nm = [fg_lc,'ST'];

n = 30;
plot(Sfd_CC(1:n),'ko'); hold on;
plot(Sfd_DD(1:n,2) + DLd(1:n,2),'rx');
plot(Ssm(1:n),'b+');
xlabel('$P_{T}$'); ylabel('$Q_{T}$'); axis equal; grid on;

lgnd = legend('True','Lin PF','Sum');
set(lgnd,'Interpreter','Latex','Location','NorthWest');

% export_fig(fig,fg_nm);
% export_fig(fig,[fg_nm,'.pdf'],'-dpdf');
%%
fig = figure('Color','White','Position',fg_ps_sht);
fg_nm = [fg_lc,'DST'];

plot(Sfd_CC-(Sfd_DD(:,2) + DLd(:,2)),'rx'); hold on;
plot(Sfd_CC-Ssm,'bx');
grid on; axis equal; 
xlabel('$\Delta P_{T}$'); ylabel('$\Delta Q_{T}$');

lgnd = legend('Lin PF','Sum');
set(lgnd,'Interpreter','Latex','Location','SouthEast');

% export_fig(fig,fg_nm);
% export_fig(fig,[fg_nm,'.pdf'],'-dpdf');
%%
fig = figure('Color','White','Position',fg_ps_sht);
fg_nm = [fg_lc,'DST_hist'];

histogram(100*abs(Sfd_CC-Ssm)./abs(Sfd_CC),'Normalization','pdf','EdgeColor','b','FaceColor','b'); hold on;
histogram(100*abs(Sfd_CC-(Sfd_DD + DLd))./abs(Sfd_CC),'Normalization','pdf','EdgeColor','r','FaceColor','r');
xlim([0,1.35]);
xlabel('$\epsilon = |\tilde{S}_{T}-S_{T}|/|\tilde{S}_{T}|\,,\, \%$');
ylabel('$p(\epsilon)$');

lgnd = legend('Lin PF','Sum');
set(lgnd,'Interpreter','Latex');
% export_fig(fig,fg_nm,'-r300');


% -------------------






%%
histogram(abs(Sfd_CC)); hold on;
histogram(abs(Sfd_DD(:,2)));
histogram(abs(Ssm));
legend('OpenDSS (True)','Linear','Sum');

%% SUMMARY STATISTICS
clc
mean(abs(Sfd_CC))
mean(abs(Sfd_DD))
mean(abs(Ssm))

var(abs(Sfd_CC))
var(abs(Sfd_DD))
var(abs(Ssm))

mean(abs(Sls_CC))
mean(abs(Sls_DD))

%%
histogram(abs(Sls_CC - Sls_DD(:,1))); hold on;
histogram(abs(Sls_CC - Sls_DD(:,2)));
histogram(abs(Sls_CC - Sls_DD(:,3)));
histogram(abs(Sls_CC - Sls_DD(:,4)));

histogram(real(Sls_CC - Sls_DD(:,1))); hold on;
histogram(real(Sls_CC - Sls_DD(:,2)));
histogram(real(Sls_CC - Sls_DD(:,3)));
histogram(real(Sls_CC - Sls_DD(:,4)));

figure
histogram(imag(Sls_CC - Sls_DD(:,1))); hold on;
histogram(imag(Sls_CC - Sls_DD(:,2)));
histogram(imag(Sls_CC - Sls_DD(:,3)));
histogram(imag(Sls_CC - Sls_DD(:,4)));

1e3*var(abs(Sls_CC))
1e3*var(abs(Sls_DD))











