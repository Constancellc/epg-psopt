clear all; close all; clc;

WD = 'C:\Users\Matt\Documents\MATLAB\epg-psopt\risk_day';
% WD = 'C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\risk_day';
cd(WD);

CC = csvread('datasets/riskDayLoadsInAndOut.csv',1,0);


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
tic % ~11 seconds x4 = ~45 seconds
for i = 1:size(Sld,1)
    sY(sYidx)=Sld(i,:);
    xhy = sparse( -1e3*[real(sY(4:end));imag(sY(4:end))] );
    
    vc = My1*xhy + a1;
    Vlin = [v0;vc];
    Slin = Vlin.*conj(Ybus_SP*Vlin)/1e3;
    Sfd_DD(i,1) = sum(Slin(1:3));
    Sls_DD(i,1) = sum(Slin);
    
    vc = My2*xhy + a2;
    Vlin = [v0;vc];
    Slin = Vlin.*conj(Ybus_SP*Vlin)/1e3;
    Sfd_DD(i,2) = sum(Slin(1:3));
    Sls_DD(i,2) = sum(Slin);
    
    vc = My3*xhy + a3;
    Vlin = [v0;vc];
    Slin = Vlin.*conj(Ybus_SP*Vlin)/1e3;
    Sfd_DD(i,3) = sum(Slin(1:3));
    Sls_DD(i,3) = sum(Slin);
    
    vc = My4*xhy + a4;
    Vlin = [v0;vc];
    Slin = Vlin.*conj(Ybus_SP*Vlin)/1e3;
    Sfd_DD(i,4) = sum(Slin(1:3));
    Sls_DD(i,4) = sum(Slin);
end
toc
%%
fn = 'compare_results';
figure;
histogram(abs(Sls_CC)); hold on;
histogram(abs(Sls_DD(:,1)));
histogram(abs(Sls_DD(:,2)));
% histogram(abs(Sls_DD(:,3)));
% histogram(abs(Sls_DD(:,4)));
% legend('PF (OpenDSS)','Lin. PF');
xlabel('$S_{l}$ (kVA)','Interpreter','Latex');
ylabel('$n$','Interpreter','Latex');
% export_fig(fig,fn);

%%
figure;
histogram(abs(Sfd_CC)); hold on;
histogram(abs(Sfd_DD(:,1)));
% histogram(abs(Sfd_DD(:,2)));
% histogram(abs(Sfd_DD(:,3)));
% histogram(abs(Sfd_DD(:,4)));
% legend('PF (OpenDSS)','Lin. PF');
xlabel('$S_{l}$ (kVA)','Interpreter','Latex');
ylabel('$n$','Interpreter','Latex');
% export_fig(fig,fn);


%%
% figure;
% histogram(abs(Sfd_CC)); hold on;
% histogram(abs(Sfd_DD));
% histogram(abs(Ssm));
% legend('OpenDSS (True)','Linear','Sum');

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











