clear all; close all; clc;

WD = 'C:\Users\Matt\Documents\MATLAB\epg-psopt\risk_day';
% WD = 'C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\risk_day';
cd(WD);

CC = csvread('riskDayLoadsInAndOut.csv',1,0);


%%
Sfd_CC = CC(:,1) + 1i*CC(:,2);
Sld = CC(:,3:2:end) + 1i*CC(:,4:2:end);
Sls_CC = Sfd_CC - sum(Sld,2);
Ssm = sum(Sld,2);
% histogram(abs(Sfd_CC)); hold on;
% histogram(abs(Ssm));
% histogram(abs(Sls_CC));


%%
load('lvtestcase_lin');
sY0 = sY(find(sY));
sYidx = find(sY);
pf = mean(real(sY0)./abs(sY0));
qK = sqrt(1-pf^2)/pf;
Ybus_SP = sparse(Ybus);
%%
Sfd_LN = zeros(size(QQ,2),1);
Sls_LN = zeros(size(QQ,2),1);
tic
for i = 1:size(QQ,2)
    sY(sYidx)=QQ(:,i);
    xhy = sparse(-1e3*[sY(4:end);sY(4:end)*qK]);
    vc = My*xhy + a;
    Vlin = [v0;vc];
    Slin = Vlin.*conj(Ybus_SP*Vlin)/1e3;
    Sfd_LN(i) = sum(Slin(1:3));
    Sls_LN(i) = sum(Slin);
end
toc
S_sum = sum(QQ,1)/pf;