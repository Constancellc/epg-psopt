close all; clear all; clc;

WD = 'C:\Users\chri3793\Documents\MATLAB\DPhil\risk_day_mtlb';
cd(WD);
%%
M = csvread('household_demand_pool_matt.csv');
figure
subplot(211)
plot(sum(M)); grid on;
subplot(212)
plot((0:0.5:23.5),sum(M));  grid on;
% ==> Biggest times: 1700-1900
%%
figure
subplot(221);
histogram(M(:,17*2 + 1))
title('1700');
subplot(222);
histogram(M(:,17.5*2 + 1))
title('1730');
subplot(223);
histogram(M(:,18*2 + 1))
title('1800');
subplot(224);
histogram(M(:,18.5*2 + 1))
title('1830');
% ==> Choose 1700 for arguments sake
%% For sake of ease, use the gamma distribution (no need to rescale)
X = M(:,17*2 + 1);
[ a,b ] = gamma_mle( X );

figure;
[N,EDGES] = histcounts(X);
x = linspace(min(EDGES),max(EDGES),1e4);
pdf = ((b^a)/gamma(a))*((x.^(a-1)).*exp(-b*x));
% figure;
histogram(X,'Normalization','pdf'); hold on;
plot(x,pdf);
% ==> seems about ok
% save('gamma_consts','a','b');