clear all; close all; clc;

WD = 'C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\risk_day';
cd(WD);

GG = load('gamma_consts');
pd = 'gamma';
nLds = 55;
nSmp = 1e4;
QQ = random(pd,GG.a,1/GG.b,nLds,nSmp);
CC = csvread('riskDayLoadIn.csv');
%%
load('lvtestcase_lin');
sY0 = sY(find(sY));
sYidx = find(sY);
pf = mean(real(sY0)./abs(sY0));
qK = sqrt(1-pf^2)/pf;
%%
Sfd = zeros(nSmp,1);
tic
for i = 1:nSmp
    sY(sYidx)=QQ(:,i);
    xhy = -1e3*[sY(4:end);sY(4:end)*qK];
    vc = My*xhy + a;
    Vlin = [v0;vc];
    Slin = Vlin.*conj(Ybus*Vlin)/1e3;
    Sfd(i) = sum(Slin(1:3));
end
toc
%%
a_sum = GG.a*numel(sYidx);
b_sum = GG.b*pf;
% S_sum = sum(QQ,1)/pf;

figure('Position',[100 200 800 500],'Color','White');
histogram(CC,'Normalization','pdf'); hold on;
histogram(abs(Sfd),'Normalization','pdf'); hold on;
% histogram(Ssm,'Normalization','pdf')

[~,EDGES] = histcounts(abs(Sfd));
x = linspace(min(EDGES),max(EDGES),1e4);
pdf = ((b_sum^a_sum)/gamma(a_sum))*((x.^(a_sum-1)).*exp(-b_sum*x));
plot(x,pdf,'k','LineWidth',2);

% legend('Linear Model','Load Sum','Analytic')
legend('OpenDSS','Linear Model','Analytic Sum')
title('Sum versus linear model PDFs (n=1e4)');
xlabel('Load');





