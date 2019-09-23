clear all; close all; %clc;
fig_loc = 'C:\Users\Matt\Documents\DPhil\thesis\c3tech1\c3figures';

set(0,'defaulttextinterpreter','latex')
set(0,'defaultaxesfontsize',12);
fig_nompos = [100 100 400 300];
%% vg_slt
fig_name = strcat(fig_loc,'\vg_slt_tss');
N = 400;
Plin = linspace(-0.8,1.1,N);
Qlin = linspace(-1.0,1.0,N);
PlinBdy = linspace(-0.8,1.1,N*5);

[P,Q] = meshgrid(Plin,Qlin);

V0 = 1.1;

STB_det = ((V0.^4)./4) + ((V0.^2).*P) - (Q.^2);

Vg2 = ((V0.^2)/2) + P + sqrt(STB_det);
Sl = ((V0.^2)/2) + P - sqrt(STB_det);


DNE = find(STB_det<=0); %does not exist
Vg2(DNE) = NaN;
Sl(DNE) = NaN;


fig = figure('Color','White','Position',fig_nompos);
[C2,H2] = contourf(P,Q,Sl,(0:0.1:1)); hold on;
[C1,H1] = contour(P,Q,sqrt(Vg2),(0.5:0.1:1.8),'LineWidth',2); hold on;
%contour(P,Q,sqrt(Vg2),(0.9:0.2:1.1),'r');

Q_bdry = sqrt(((V0.^4)./4) + ((V0.^2).*PlinBdy));
stb_det = ((V0.^4)./4) + ((V0.^2).*PlinBdy);
% dne = find(stb_det<=0);
dne = find(stb_det>0);
% Q_bdry(dne) = NaN;
Q_bdry = Q_bdry(dne);
PlinBdy = PlinBdy(dne);

Q_bdry = [-Q_bdry(end:-1:1) , Q_bdry];
PlinBdy = [PlinBdy(end:-1:1) , PlinBdy ];
plot(PlinBdy,Q_bdry,'k')
plot(PlinBdy,-Q_bdry,'k')

xlabel('$p_{\mathrm{Snd}}$')
ylabel('$q_{\mathrm{Snd}}$')
% title('$|V_{g}|$ and $\tilde{S}_{l}$ against ($\tilde{P_{g}}$,$\tilde{Q_{g}}$)');
axis('equal'); %grid on;
axis([min(Plin) max(Plin) min(Qlin) max(Qlin)]);

% lgnd = legend([H1,H2],'$|V_{g}|$','$\tilde{S}_{l}$','Location','NorthWest');
lgnd = legend([H1,H2],'$|V_{\mathrm{Snd}}|$','$s_{\mathrm{lss}}$','Location','NorthWest');
set(lgnd,'FontSize',12,'Interpreter','Latex');

export_fig(gcf,fig_name,'-m3');  %,'dpi',300
close;
% export_fig(gcf,strcat(fig_name,'.pdf'),'-dpdf');
