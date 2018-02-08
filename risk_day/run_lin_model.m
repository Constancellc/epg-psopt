clear all; close all; clc;
% I think this is the old code that was used hat was superceded by the
% 'compare_results.m' script that takes the direct loads and simply applies
% them straight to the problem.

WD = 'C:\Users\Matt\Documents\MATLAB\epg-psopt\risk_day';
% WD = 'C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\risk_day';
cd(WD);

GG = load('gamma_consts');
pd = 'gamma';
nLds = 55;
nSmp = 144000;

% QQ = random(pd,GG.a,1/GG.b,nLds,nSmp); %NB requires statistics toolbox
QQ = csvread('riskDaySpecifiedVsAppliedLoaad.csv',1,1);
QQ = reshape(QQ,nLds,numel(QQ)/nLds);
CC = csvread('riskDayLoadIn.csv');
%%
load('lvtestcase_lin');
sY0 = sY(find(sY));
sYidx = find(sY);
pf = mean(real(sY0)./abs(sY0));
qK = sqrt(1-pf^2)/pf;
Ybus_SP = sparse(Ybus);
%%
Sfd = zeros(size(QQ,2),1);
Sl = zeros(size(QQ,2),1);
tic
for i = 1:size(QQ,2)
    sY(sYidx)=QQ(:,i);
    xhy = sparse(-1e3*[sY(4:end);sY(4:end)*qK]);
    vc = My*xhy + a;
    Vlin = [v0;vc];
    Slin = Vlin.*conj(Ybus_SP*Vlin)/1e3;
    Sfd(i) = sum(Slin(1:3));
    Sl(i) = sum(Slin);
end
toc
S_sum = sum(QQ,1)/pf;
%%
Sl_OD = CC' - sum(QQ,1);
histogram(real(Sl)); hold on;
% histogram(Sl_OD)

%%
subplot(211);  
histogram(CC');
subplot(212)
histogram(sum(QQ,1));
%%
fgnm = 'run_lin_model';
fig = figure('Position',[100 200 500 300],'Color','White');
% figure('Position',[100 200 800 500],'Color','White');

HH = histogram(CC/pf,'Normalization','pdf'); hold on;
HH = histogram(abs(Sfd),HH.BinEdges,'Normalization','pdf'); hold on;
HH = histogram(S_sum,HH.BinEdges,'Normalization','pdf'); hold on;
% x = linspace(min(HH.BinEdges),max(HH.BinEdges),1e4);
% a_sum = GG.a*numel(sYidx);
% b_sum = GG.b*pf;
% pdf = ((b_sum^a_sum)/gamma(a_sum))*((x.^(a_sum-1)).*exp(-b_sum*x));
% plot(x,pdf,'k','LineWidth',2);
% legend('(i) Full Load Flow','(ii) Linear Load Flow','(iii) No Load Flow')
legend('(i) True (OpenDSS)','(ii) Linear','(iii) Load Sum')
% title('Sum versus linear model PDFs (n=1e4)');
xlabel('S_T_r (kVA)'); ylabel('P(S_T_r)');

% export_fig(fig,fgnm);

%%
clc
m1 = mean(CC)/pf
m2 = mean(abs(Sfd))
m3 = mean(S_sum)

v1 = var(CC/pf)
v2 = var(abs(Sfd))
v3 = var(S_sum)

em2 = (m2-m1)/m1
em3 = (m3-m1)/m1






