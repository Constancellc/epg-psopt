clear all

FD = 'C:\Users\Matt\Documents\DPhil\pesgm19\pesgm19_paper\figures\';
set(0,'defaulttextinterpreter','Latex');

fig = figure('Color','White','Position',[100 150 500 270]);
% fig = figure('Color','White','Position',[100 150 350 270]);

Vp = 1.10;

rng('default')

% FN = [FD,'full'];
% x = linspace(0,1,4000);

% y = 1.06 + 0.12*x + x.*0.02.*randn(size(x));
% plot(x,y,'x')
% hold on;
% plot([0,1],[1,1]*Vp,'k--')

% plot(0.23*[1,1],[1.05,1.25],'g:','Linewidth',2)
% plot(0.72*[1,1],[1.05,1.25],'r:','Linewidth',2)

% text(0.17,1.16,sprintf('Min host.\n capacity'),'Rotation',90)
% text(0.66,1.19,sprintf('Max host.\n capacity'),'Rotation',90)
% text(0.015,1.107,'$v^{+}$');

FN = [FD,'methods'];
subplot(121);
nx = 4;
nX = 1e3;

x = linspace(1/nx,1,nx);
X = ones(nX,1)*x;
Y = 1.06 + 0.1*X + X.*0.02.*randn(size(X));

boxplot(Y,x,'whisker',10); hold on;
plot([0,11],[1,1]*Vp,'k--')

xlabel('Power');
ylabel('Voltage');

subplot(122);
y = Vp;
x = 0.26 + 0.08*abs(randn(1e3,1));

boxplot(x,y,'orientation','horizontal','widths',0.5,'whisker',10); hold on;
plot([0,1],[1,1],'k--');
axis([0,1,0,4]);
yticks(linspace(0,4,9));
yticklabels({'1.06','1.08','1.1','1.12','1.14','1.16','1.18','1.2','1.22'})

xlabel('Power');
ylabel('Voltage');


export_fig(fig,FN);
export_fig(fig,[FN,'.pdf'],'-dpdf');
saveas(fig,FN,'meta')