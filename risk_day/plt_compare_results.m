clear all; close all; clc;

WD = 'C:\Users\Matt\Documents\MATLAB\epg-psopt\risk_day';
% WD = 'C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\risk_day';
cd(WD);

fg_lc = 'C:\Users\Matt\Documents\DPhil\risk_day\rd_ppt\figures\';
% fg_lc = 'C:\Users\chri3793\Documents\DPhil\risk_day\rd_ppt\figures\';   
fg_ps = [200 300 400 400];

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

DD = load('datasets/lvtestcase_lin1');
a1 = DD.a1; My1 = DD.My1;
DD = load('datasets/lvtestcase_lin2');
a2 = DD.a2; My2 = DD.My2;
DD = load('datasets/lvtestcase_lin3');
a3 = DD.a3; My3 = DD.My3;
DD = load('datasets/lvtestcase_lin4');
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
tic % ~11 seconds x4 = ~45 seconds
for i = 1:size(Sld,1)

    sY(sYidx)=Sld(i,:);
    xhy = sparse( -1e3*[real(sY(4:end));imag(sY(4:end))] );

    vc = My1*xhy + a1;
    Vlin = [v0;vc];
    Slin = Vlin.*conj(Ybus_SP*Vlin)/1e3;
    Sfd_DD(i,1) = sum(Slin(1:3));
    Sld_DD(i,1) = sum(Slin(sYidx));
%     Sls_DD(i,1) = sum(Slin);
    

    vc = My2*xhy + a2;
    Vlin = [v0;vc];
    Slin = Vlin.*conj(Ybus_SP*Vlin)/1e3;
    Sfd_DD(i,2) = sum(Slin(1:3));
    Sld_DD(i,2) = sum(Slin(sYidx));
%     Sls_DD(i,2) = sum(Slin);
    
    vc = My3*xhy + a3;
    Vlin = [v0;vc];
    Slin = Vlin.*conj(Ybus_SP*Vlin)/1e3;
    Sfd_DD(i,3) = sum(Slin(1:3));
    Sld_DD(i,3) = sum(Slin(sYidx));
%     Sls_DD(i,3) = sum(Slin);
    
    vc = My4*xhy + a4;
    Vlin = [v0;vc];
    Slin = Vlin.*conj(Ybus_SP*Vlin)/1e3;
    Sfd_DD(i,4) = sum(Slin(1:3));
    Sld_DD(i,4) = sum(Slin(sYidx));
%     Sls_DD(i,4) = sum(Slin);
end
toc

DLd = Sld_DD + Ssm;
Sfd_DD = Sfd - DLd;
Sls_DD = Sfd_DD + Sld_DD;

%%
n = 1440;
%
figure;
plot(Sfd_CC(1:n),'ko'); hold on;
plot(Sfd_DD(1:n,1),'rx');
plot(Sfd_DD(1:n,2),'bx');
plot(Sfd_DD(1:n,3),'gx');
plot(Sfd_DD(1:n,4),'kx');
axis equal;
xlabel('P'); ylabel('Q'); grid on;

%%
plot(DLd,'x')


%%





%%
fig = figure('Color','White','Position',fg_ps);
fg_nm = [fg_lc,'Sl'];

n=80;
plot(Sls_CC(1:n),'ko'); hold on;
plot(Sls_DD(1:n,:),'x');
axis equal;
xlabel('$P_{l}$'); ylabel('$Q_{l}$'); grid on;
lgnd = legend('True','$0.2\,\hat{S}$','$0.6\,\hat{S}$','$1.0\,\hat{S}$','$1.4\,\hat{S}$');
set(lgnd,'Location','NorthWest','Interpreter','Latex');

% export_fig(fig,fg_nm);
% export_fig(fig,[fg_nm,'.pdf'],'-dpdf');

%%
fig = figure('Color','White','Position',fg_ps);
fg_nm = [fg_lc,'DSl'];

plot(Sls_DD - Sls_CC,'x')

xlabel('$\Delta P_{l}$'); ylabel('$\Delta Q_{l}$'); grid on; axis equal;
lgnd = legend('$0.2\,\hat{S}$','$0.6\,\hat{S}$','$1.0\,\hat{S}$','$1.4\,\hat{S}$');
set(lgnd,'Location','NorthWest','Interpreter','Latex');

% export_fig(fig,fg_nm);
% export_fig(fig,[fg_nm,'.pdf'],'-dpdf');

%%
n = 20;
figure;
plot(Sfd_CC(1:n),'ko'); hold on;
plot(Sfd_DD(1:n,:) + DLd(1:n,:),'x');


%%
figure;
plot(Sfd_CC-(Sfd_DD + DLd),'x');
grid on; axis equal; 
xlabel('DPT'); ylabel('DQT');



%%

histogram(abs(Sfd_CC)); hold on;
histogram(abs(Sfd_DD));
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











