close all; clear all; clc;

% WD = 'C:\Users\chri3793\Documents\MATLAB\DPhil\risk_day_mtlb';
WD = 'C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\risk_day';
cd(WD);

fg_lc = 'C:\Users\chri3793\Documents\DPhil\risk_day\rd_ppt\figures\';
fg_ps_tll = [200 300 340 400];
fg_ps = [200 300 400 400];


set(0,'defaulttextinterpreter','latex');
set(0,'defaultaxesfontsize',14);
set(0,'defaulttextfontsize',14);
%%
M = csvread('household_demand_pool_matt.csv');

%%
fg_nm = [fg_lc,'all_loads'];
fig = figure('Color','White','Position',fg_ps_tll);

plot((0:0.5:23.5),M');  grid on;
xlabel('Time (hour)'); xlim([0 24]);
ylabel('Power (kW)'); xticks((0:4:24))

% export_fig(fig,fg_nm);
% export_fig(fig,[fg_nm,'.pdf'],'-dpdf');
%%
fg_nm = [fg_lc,'all_loads_sum'];
fig = figure('Color','White','Position',fg_ps_tll);

plot((0:0.5:23.5),sum(M));  grid on;
xlabel('Time (hour)'); xlim([0 24]);
ylabel('Power (kW)'); xticks((0:4:24))
ylim([0 300]);

% export_fig(fig,fg_nm);
% export_fig(fig,[fg_nm,'.pdf'],'-dpdf');
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
fg_nm = [fg_lc,'mle_500'];
fig = figure('Position',fg_ps,'Color','White');

X = M(:,17*2 + 1);

X = X+1e-6; % get rid of any zeros
xb = mean(X);
A = logspace(-1,1,1e4);
B = A/xb;
lL = sum( ones(size(X))*(A.*log(B) - log(gamma(A))) + ...
                        log(X)*(A - 1) - X.*B , 1 );
semilogx(A,lL);
xlabel('Shape parameter $\alpha$');

export_fig(fig,fg_nm);
export_fig(fig,[fg_nm,'.pdf'],'-dpdf');
%%
fg_nm = [fg_lc,'nom_histogram'];
fig = figure('Position',fg_ps,'Color','White');

HH = histogram(X,'Normalization','pdf');
xlabel('Household Load $P_{H}$, (kW)'); 
ylabel('$p(P_{H})$');
lgnd = legend('$n = 500$ houses'); set(lgnd,'Interpreter','Latex');

% export_fig(fig,fg_nm,'-r300');
%%
[ a,b ] = gamma_mle( X );

fg_nm = [fg_lc,'find_distrb'];
fig = figure('Position',fg_ps,'Color','White');

% figure;
HH = histogram(X,'Normalization','pdf'); hold on;
x = linspace(min(HH.BinEdges),max(HH.BinEdges),1e4);
pdf = ((b^a)/gamma(a))*((x.^(a-1)).*exp(-b*x));

plot(x,pdf,'k','Linewidth',2);
xlabel('Household Load $P_{H}$, (kW)'); 
ylabel('$p(P_{H})$');
legend('Smart Meter Data (17:00)','Fitted Gamma PDF');
export_fig(fig,fg_nm,'-r300');
% ==> seems about ok
% save('gamma_consts','a','b');

%%
fig = figure('Color','White','Position',fg_ps);
fg_nm = [fg_lc,'gamma_ab'];

aa = NaN*zeros(5,93);
bb = NaN*zeros(5,93);
for i = 1:5
    Xa = X((i-1)*93 + 1:i*93);
    for j = 1:91
        [ aa(i,j),bb(i,j) ] = gamma_mle( Xa(j:end) );
    end
end
plot((ones(5,1)*(93:-1:1))',aa');
xlabel('No. Houses'); ylabel('$\alpha$');
ylim([0 4]); xticks((0:20:100)); grid on;

% export_fig(fig,fg_nm);
% export_fig(fig,[fg_nm,'.pdf'],'-dpdf');
%%
HH = histogram(X,'Normalization','pdf'); hold on;
x = linspace(min(HH.BinEdges),max(HH.BinEdges),1e4); close;

fig = figure('Color','White','Position',fg_ps);
fg_nm = [fg_lc,'pdfs'];
for i = 1:size(aa,1)
    a = aa(i,1);
    b = bb(i,1);
    pdf = ((b^a)/gamma(a))*((x.^(a-1)).*exp(-b*x));
    plot(x,pdf); hold on;
end
ylim([0 2]); grid on;

xlabel('Household Load $P_{H}$, (kW)'); 
ylabel('$p(P_{H})$');
% export_fig(fig,fg_nm);
% export_fig(fig,[fg_nm,'.pdf'],'-dpdf');





