% derived optima is a script that considers and plots the derived optima
% from 12/07.

clear all; close all; 
fig_loc = 'C:\Users\Matt\Documents\DPhil\thesis\c3tech1\c3figures';

set(0,'defaulttextinterpreter','latex');
set(0,'defaultaxesfontsize',12);
fig_nompos = [100 100 550 320];

%%
% Vg = 1.1;
Vg = 1.0;

% v0_lin = linspace(0.9,1.2,300);
v0_lin = linspace(0.85,1.15,300);

% theta = linspace(0.003,(pi/2) - 0.0006,1000);
theta = linspace(0.0099,(pi/2) - 0.0099,1000);


lz_lin = cot(theta); % lambda
lv_lin = Vg./v0_lin;
nu_lin = 1./lv_lin;

%[lz,lv] = meshgrid(lz_lin,lv_lin);
[lz,nu] = meshgrid(lz_lin,nu_lin);
lv = 1./nu;
V0 = Vg./lv;

lr = lz./sqrt(1 + (lz.^2));
lx = 1./sqrt(1 + (lz.^2));
Z = lr + 1i*lx;

Pgt = (V0.^2).*lv.*(lv - lr); % we know it is the two negative solutions from the derivation
Qgt = -(V0.^2).*lv.*lx;

Sl = (Pgt.^2 + Qgt.^2)./(Vg.*(lr - 1i*lx));

P0pr = (V0.^2).*lr.*((lv.^2) - 1) - (lx.*(lz.*Pgt + Qgt)); %NB |Z| = 1!
Q0pr = (V0.^2).*lx.*(lv.^2 - (2.*lr.*lv));
Pgpr = lr.*Pgt - lx.*Qgt;
Qgpr = lx.*Pgt + lr.*Qgt;

PF0pr = abs(P0pr)./sqrt(P0pr.^2 + Q0pr.^2).*(P0pr>=0);
PFgpr = abs(Pgpr)./sqrt(Pgpr.^2 + Qgpr.^2).*(P0pr>=0);
PF0pr = abs(P0pr)./sqrt(P0pr.^2 + Q0pr.^2).*(P0pr>=0);
PFgpr = abs(Pgpr)./sqrt(Pgpr.^2 + Qgpr.^2).*(P0pr>=0);


% stability values:
P0stb = (V0.^2).*( (-0.5*lr) + (lx.*sqrt((lv.^2) - 0.25)) );
Pgstb = P0stb + (((V0.*lv).^2).*lr);
Slstb = (Vg.^2)./conj(Z);

P0prpr = P0stb.*(Pgpr>Pgstb) + P0pr.*(Pgpr<Pgstb);
Slprpr = Slstb.*(Pgpr>Pgstb) + Sl.*(Pgpr<Pgstb);

% eta = (P0pr./(P0pr + real(Slprpr))).*(P0pr>=0) ... 
%                                     + (P0pr<0)*-0.001; %efficiency
eta = (P0pr./Pgpr).*(P0pr>=0) ... 
                                    + (P0pr<0)*-0.001; %efficiency
% subplot(3,2,3)
% [c_con,~] = contour(log10(lz),lv,P0stb,(-2:0.1:2));
% clabel(c_con)
% xlabel('log10(Lz[= R/X]) '); ylabel('Lv = Vg/V0');
% legend('P0stb');
% grid on;


%subplot(3,2,1)
% mgtt_p0
fig_name = strcat(fig_loc,'\mgtt_p0');

figure('Color','White','Position',fig_nompos);
[c_con,~] = contour(lz,V0,P0pr,(-0.1:0.1:2));
set(gca,'xscale','log');
clabel(c_con)
xlabel('$R/X$ ratio, $\lambda$ '); ylabel('Source voltage, $V_{\mathrm{Rcv}}$, pu');
%title('Maximum Generation Transfer ($|V_{g}| = 1.1$ pu)');

lgnd = legend('$P_{\mathrm{Rcv,\,MPT}}$','Location','SouthEast');
set(lgnd,'FontSize',12,'Interpreter','Latex');
grid on;

% export_fig(gcf,fig_name);
% export_fig(gcf,strcat(fig_name,'.pdf'),'-dpdf');
%%
fig_name = strcat(fig_loc,'\mgtt_p0_stb_tss');

figure('Color','White','Position',fig_nompos);
[c_con,~] = contour(lz,V0,P0prpr,(-0.1:0.1:2));
set(gca,'xscale','log');
clabel(c_con)

xlabel('$R/X$ ratio, $\lambda$ '); ylabel('Source voltage, $V_{\mathrm{Rcv}}$, pu');
title('Maximum Generation Transfer ($|V_{g}| = 1.1$ pu)');

lgnd = legend('$P_{\mathrm{Rcv,\,MPT}}$');
set(lgnd,'FontSize',12,'Interpreter','Latex');
grid on;

% export_fig(gcf,fig_name);
% export_fig(gcf,strcat(fig_name,'.pdf'),'-dpdf');

%% mgtt_eta
figure('Color','White','Position',fig_nompos);
fig_name = strcat(fig_loc,'\mgtt_eta_tss');
[c_con,~] = contour(lz,V0,eta,(0:0.1:1)); hold on;
[~,~] = contour(lz,V0,P0pr,[-1,1]*1e-6,'k','linewidth',1.5);

clabel(c_con,[0 0.2 0.4 0.6 0.8 0.9]); grid on;

% [~,~] = contour(lz,V0,P0pr,[-0.001,0.001],'k');
set(gca,'xscale','log');

xlabel('$R/X$ ratio, $\lambda$ '); ylabel('Source voltage, $V_{\mathrm{Rcv}}$, pu');
% title('Efficiency $\eta$ of power transmission ($|V_{g}| = 1.1$ pu)'); 

lgnd = legend('$\frac{P_{\mathrm{Rcv,\,MPT}}}{P_{\mathrm{Snd,\,MPT}}}$ ','Location','NorthWest');
set(lgnd,'FontSize',18,'Interpreter','Latex');

% export_fig(gcf,fig_name);
% export_fig(gcf,strcat(fig_name,'.pdf'),'-dpdf'); close;
%% mgtt_pf_g
figure('Color','White','Position',fig_nompos);
fig_name = strcat(fig_loc,'\mgtt_pf_g_tss');
[c_con,~] = contour(lz,V0,PFgpr); hold on;
[~,~] = contour(lz,V0,P0pr,[-1,1]*1e-6,'k','linewidth',1.5);

set(gca,'xscale','log');
clabel(c_con,[0.1 0.3 0.5 0.7 0.8 0.9]); grid on;

xlabel('$R/X$ ratio, $\lambda$ '); ylabel('Source voltage, $V_{\mathrm{Rcv}}$, pu');
%lgnd = legend('$/$');
lgnd = legend('$\frac{P_{\mathrm{Snd,\,MPT}}}{|S_{\mathrm{Snd,\,MPT}}|}$');
set(lgnd,'FontSize',18,'Interpreter','Latex','Location','NorthWest');
%title('Generator power factor at maximum power transfer ($|V_{g}| = 1.1$)'); 

% export_fig(gcf,fig_name);
% export_fig(gcf,strcat(fig_name,'.pdf'),'-dpdf'); close;
%% mgtt_pf_0
figure('Color','White','Position',fig_nompos);
fig_name = strcat(fig_loc,'\mgtt_pf_0_tss');
[c_con,~] = contour(lz,V0,PF0pr); hold on;
[~,~] = contour(lz,V0,P0pr,[-1,1]*1e-6,'k','linewidth',1.5);

set(gca,'xscale','log');
clabel(c_con,[0.1 0.3 0.5 0.7 0.9]); grid on;

xlabel('$R/X$ ratio, $\lambda$ '); ylabel('Source voltage, $V_{\mathrm{Rcv}}$, pu');
%lgnd = legend('$/$');
lgnd = legend('$\frac{P_{\mathrm{Rcv,\,MPT}}}{|S_{\mathrm{Rcv,\,MPT}}|}$');
set(lgnd,'FontSize',18,'Interpreter','Latex','Location','NorthWest');
%title('Generator power factor at maximum power transfer ($|V_{g}| = 1.1$)'); 

% export_fig(gcf,fig_name);
% export_fig(gcf,strcat(fig_name,'.pdf'),'-dpdf'); close;











