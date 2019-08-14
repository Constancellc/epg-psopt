close all; clear all

fig_loc = 'C:\Users\Matt\Documents\DPhil\thesis\c3tech1\c3figures';

% set(0,'defaulttextinterpreter','latex');
% set(0,'defaultaxesfontsize',12);
fig_nompos = [100 100 550 320];
%%
V0 = 1;
% Vg = 1.1;
Vg = 1.05;
% Vlo = 0.9;

N = 1e4;

theta_lo = 0.01;
theta_hi = pi/2 - 0.01;
theta = linspace(theta_lo,theta_hi,N);
lz = cot(theta);
RX = lz;
Z = cos(theta) + 1i*sin(theta);


lr = lz./sqrt(1 + (lz.^2));
lx = 1./sqrt(1 + (lz.^2));
lv = Vg./V0;
nu = 1./lv;

Pgt = (V0.^2).*lv.*(lv - lr); % we know it is the two negative solutions from the derivation
Qgt = -(V0.^2).*lv.*lx;

Sl = (Pgt.^2 + Qgt.^2)./(Vg.*(lr - 1i*lx));

P0pr = (V0.^2).*lr.*((lv.^2) - 1) - (lx.*(lz.*Pgt + Qgt)); %NB |Z| = 1!
Pgpr = lr.*Pgt - lx.*Qgt;
Qgpr = lx.*Pgt + lr.*Qgt;


% stability values:
P0stb = (V0.^2).*( (-0.5*lr) + (lx.*sqrt((lv.^2) - 0.25)) );
Pgstb = P0stb + (((V0.*lv).^2).*lr);
Slstb = (Vg.^2)./conj(Z);

% stab/volt values:
lz_prpr = 1./sqrt((2*lv)^2 - 1);

Pgprpr = Pgstb.*(Pgpr>Pgstb) + Pgpr.*(Pgpr<Pgstb);
P0prpr = P0stb.*(Pgpr>Pgstb) + P0pr.*(Pgpr<Pgstb);
Slprpr = Slstb.*(Pgpr>Pgstb) + Sl.*(Pgpr<Pgstb);

% Unity PF values:
lz_0 = sqrt((lv^2) - 1);

Pgupf = (V0.^2).^lv.*( (lr.*lv) - sqrt(1 - ((lx.*lv).^2) ) );
P0upf = Pgupf.*(1 - ((lr.*Pgupf)./(Vg.^2)));

Pgupf = Pgupf./(1-(lz<lz_0)); %NaN if doesn't exist
P0upf = P0upf./(1-(lz<lz_0));

%load('datasets/loss_voltage_limit_SgSnresults.mat','Pg','Pn','Qg','Qn','RX','Z','theta');
%% mgtt_1dcomparison
fig_name = strcat(fig_loc,'\mgtt_1dcomparison_tss');
fig = figure('Color','White','Position',fig_nompos); 

semilogx(lz,Pgpr,'k--'); hold on;
semilogx(lz,P0pr,'k');
% semilogx(lz,Pgprpr,'--','Color',0.6*[1 1 1]);
% semilogx(lz,P0prpr,'Color',0.6*[1 1 1]);
semilogx(lz,Pgstb,'r--'); 
semilogx(lz,P0stb,'r');
semilogx(lz,real(Pgupf),'b--'); %avoid small imaginary residuals
semilogx(lz,real(P0upf),'b'); %avoid small imaginary residuals

grid on;
xlabel('$R/X$ ratio, $\lambda$');
ylabel('Line/substation power $P_{(\cdot)}$, p.u.');
%title('Analytic versus Heuristic maximum generation power transfer');

%lgnd = legend('$P_{\mathrm{Snd: MPT}}$','$P_{\mathrm{Sub,\,MPT}}$','$P_{\mathrm{Snd: MPT,\,MPT}}$','$P_{\mathrm{Sub,\,MPT,\,MPT}}$',...
%              '$P_{\mathrm{Snd,\,Stb}}$','$P_{\mathrm{Sub,\,Stb}}$','$P_{\mathrm{Snd,\,UPF}}$','$P_{\mathrm{Sub,\,UPF}}$');
lgnd = legend('$P_{\mathrm{Snd,\,MPT}}$','$P_{\mathrm{Sub,\,MPT}}$',...
              '$P_{\mathrm{Snd,\,Stb}}$','$P_{\mathrm{Sub,\,Stb}}$','$P_{\mathrm{Snd,\,UPF}}$','$P_{\mathrm{Sub,\,UPF}}$');

set(lgnd,'Interpreter','latex','Location','SouthWest','fontsize',13);


% export_fig(gcf,fig_name);
% export_fig(gcf,strcat(fig_name,'.pdf'),'-dpdf'); close;


% V0/sqrt( (4*(Vg^2)) - (V0^2))












