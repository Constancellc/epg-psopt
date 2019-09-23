clear all; close all;
% addpath('psnov18_fncs');
fig_loc = 'C:\Users\Matt\Documents\DPhil\thesis\c3tech1\c3figures\';
set(0,'defaulttextinterpreter','latex'); set(0,'defaultaxesfontsize',14);
fig_nompos = [600 200 550 300];
%% First create some solar data:

% delta = 23.65*pi/180; % july day
% 
% atan(cot(delta)/1e-3)*180/pi % North pole
% atan(cot(delta)/1)*180/pi % Lulea, Fairbanks, Reykjavik
% atan(cot(delta)/2)*180/pi % Paris, Winnipeg
% atan(cot(delta)/4)*180/pi % Houston, New Dehli
% atan(cot(delta)/1e3)*180/pi % Nairobi, Singapore
t = (-12:1/(60):12-1/(60));
w = pi/12; %rad/hour

k = 2;
Ps_pk = 1.0;

Ps = Ps_pk*(1 + k*cos(w*t))/(1+k);
Ps = Ps.*(Ps>0);

% save('solar_curves_dataset.mat','Ps','t'); % to create the dataset
%%
Pgp = 1.08;
Pgh = 1.0;
Pgd = 0.75;
Pgb = 0.45;

eps = 3e-3;

% Ps_d = min(Ps,Pgd);
% Ps_b = min(Ps,Pgb);

Ps_b = min([Ps+eps; Pgb*ones(size(Ps))]);
Ps_d = min([Ps+2*eps; Pgd*ones(size(Ps))]);


Pshade00 = zeros(size(Ps))./(Ps~=0);
Pshade01 = min( [Ps;Pgb*ones(size(Ps))] ,[],1)./(Ps~=0);
Pshade10 = (Pgb)*ones(size(Ps))./(Ps>Pgb);
Pshade11 = min( [Ps;Pgd*ones(size(Ps))] ,[],1)./(Ps>Pgb);

fig_name = [fig_loc,'solar_curves_tss'];
fig = figure('Color','White','Position',fig_nompos);

pp=area(t+12,[Pshade01',(Pshade11-Pshade01)'],'EdgeColor','White'); hold on;
% set(pp(1),'FaceColor',[0.8 0.98 0.8]);
% set(pp(2),'FaceColor',[0.8 0.8 0.98]);
set(pp(1),'FaceColor',[0.7 0.98 0.7]);
set(pp(2),'FaceColor',[0.5 0.7 0.98]);

% pp(3) = plot(t+12,Ps,'Linewidth',2,'Color',[0.6 0.3 0.5]); hold on;
pp(3) = plot(t+12,Ps,'Linewidth',1,'Color',rgb('orangered')); hold on;

% pp(7) = plot(t+12,Ps_d,'b-.','Linewidth',2); hold on;
pp(7) = plot(t+12,Ps_d,'b','Linewidth',2); hold on;
% pp(6) = plot(t+12,Ps_b,'g--','Linewidth',2); hold on;
pp(6) = plot(t+12,Ps_b,'g','Linewidth',1); hold on;


xs = axis;
pp(4) = plot(xs(1:2),Pgd*[1 1],'k-.');
pp(5) = plot(xs(1:2),Pgb*[1 1],'k:');

% text(-2.5,Pgb,'$P_{\mathrm{gen}}^{\mathrm{base}}$','Interpreter','Latex','FontSize',14);
% text(-2.5,Pgd,'$P_{\mathrm{gen}}^{\mathrm{chng}}$','Interpreter','Latex','FontSize',14);
text(0.15,Pgb+0.08,'$\hat{P}_{\mathrm{gen}}^{\mathrm{base}}$','Interpreter','Latex','FontSize',14);
text(0.15,Pgd+0.08,'$\hat{P}_{\mathrm{gen}}^{\mathrm{chng}}$','Interpreter','Latex','FontSize',14);

xlabel('Time, $\tau $ (hour)');
ylbl=ylabel('Power, $P_{(\cdot)}$, pu');

lgnd = legend([pp(3),pp(1),pp(2)],'$P_{\mathrm{ipt}}$','$E_{\mathrm{gen}}^{\mathrm{base}}$',...
    '$\Delta E_{\mathrm{gen}}$');
set(lgnd,'Interpreter','Latex','FontSize',14);
axis([0 24 0 1.3]);

xticks([0 6 12 18 24]);
yticks([0 1]);
grid off;

% set(ylbl,'Position',ylbl.Position.*[1.15 1.2 1.0])

x = [0.18 0.18];
y = [0.44,0.61];

% annotation('textarrow',x,y);
% annotation('textarrow',x,circshift(y,1));
% text(2,0.59,'$\Delta P_{\mathrm{g}}^{\mathrm{Lim}}$','Interpreter','Latex','Fontsize',16);

export_fig(fig,fig_name,'-r400'); close;
% export_fig(fig,[fig_name,'.pdf'],'-dpdf');
% export_fig(fig,[fig_name,'.eps'],'-deps');



