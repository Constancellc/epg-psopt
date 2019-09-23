clear all

% FD = 'C:\Users\Matt\Documents\DPhil\pesgm19\pesgm19_paper\figures\';
FD = 'C:\Users\Matt\Documents\DPhil\papers\pesgm19\pesgm19_poster\figures\';
set(0,'defaulttextinterpreter','Latex');
colorOrder = get(0, 'DefaultAxesColorOrder');

pltFull = 0;
pltMethods = 1;
pltPoster = 1;
pltPres = 1; % also requires pltPoster.
pltThesis = 0;
exportFig = 1;

Vp = 1.10;

rng('default')

if pltFull
    fig = figure('Color','White','Position',[100 150 500 270]);
    FN = [FD,'full'];
    x = linspace(0,1,4000);

    y = 1.06 + 0.12*x + x.*0.02.*randn(size(x));
    plot(x,y,'x')
    hold on;
    plot([0,1],[1,1]*Vp,'k--')

    plot(0.23*[1,1],[1.05,1.25],'g:','Linewidth',2)
    plot(0.72*[1,1],[1.05,1.25],'r:','Linewidth',2)

    text(0.17,1.16,sprintf('Min host.\n capacity'),'Rotation',90)
    text(0.66,1.19,sprintf('Max host.\n capacity'),'Rotation',90)
    text(0.015,1.107,'$v^{+}$');
end

if pltMethods && pltPoster==0 && pltThesis==0
    fig = figure('Color','White','Position',[100 150 500 270]);
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
end

if pltMethods && pltPoster
    fig = figure('Position',[100 150 250 400]);
    FN = [FD,'methodsPstr'];
    subplot(211);
    nx = 4;
    nX = 1e3;

    x = linspace(1/nx,1,nx);
    X = ones(nX,1)*x;
    Y = 1.06 + 0.1*X + X.*0.02.*randn(size(X));

    boxplot(Y,x,'whisker',10); hold on;
    plot([0,11],[1,1]*Vp,'k--')
    xlabel('Power');
    ylabel('Voltage');

    subplot(212);
    y = Vp;
    x = 0.26 + 0.08*abs(randn(1e3,1));

    boxplot(x,y,'orientation','horizontal','widths',0.5,'whisker',10); hold on;
    xlims = [0.15,1.1];
    plot(xlims,[1,1],'k--');
    axis([xlims,0,4]);
    yticks(linspace(0,4,9));
    yticklabels({'1.06','1.08','1.1','1.12','1.14','1.16','1.18','1.2','1.22'})
    yticks([1,2.25,3.5]);
    yticklabels({'1.1','1.15','1.2'})
    
    xticks([0.25,0.5,0.75,1.0])

    xlabel('Power');
    ylabel('Voltage');
end


if pltMethods && pltThesis
    PositionSet = [100 150 600 210];
    fig = figure('Position',PositionSet);
%     FN = [FD,''];
    FN = ['C:\Users\',getenv('username'),'\Documents\DPhil\thesis\c4tech2\c4figures\methodsThss'];
    subplot(121);
    nx = 4;
    nX = 1e3;

    x = linspace(1/nx,1,nx);
    X = ones(nX,1)*x;
    Y = 1.06 + 0.1*X + X.*0.02.*randn(size(X));

    boxplot(Y,x,'whisker',10,'color',colorOrder(1,:)); hold on;
    plot([0,11],[1,1]*Vp,'k:')
    xlabel('Power');
    ylabel('Voltage');
    
    xs = axis;
    dP = 0.2*xs(4)/PositionSet(4); % not sure why 0.2*?
    MSize = 5;
    plot([1,2,3,4],max(Y) + dP*MSize,'v','color',colorOrder(1,:),'MarkerFaceColor','w','MarkerSize',MSize);
    plot([1,2,3,4],min(Y) - dP*MSize,'^','color',colorOrder(1,:),'MarkerFaceColor','w','MarkerSize',MSize);
    
    subplot(122);
    y = Vp;
    x = 0.26 + 0.08*abs(randn(1e3,1));

    boxplot(x,y,'orientation','horizontal','widths',0.5,'whisker',10,'color',colorOrder(1,:)); hold on;
    xlims = [0.15,1.1];
    plot(xlims,[1,1],'k:');
    axis([xlims,0,4]);
    yticks(linspace(0,4,9));
    yticklabels({'1.06','1.08','1.1','1.12','1.14','1.16','1.18','1.2','1.22'})
    yticks([1,2.25,3.5]);
    yticklabels({'1.1','1.15','1.2'})
    
    xticks([0.25,0.5,0.75,1.0])

    xs = axis;
    dP = xs(2)/PositionSet(2);
    MSize = 5;
    plot(max(x) + dP*MSize,1,'<','color',colorOrder(1,:),'MarkerFaceColor','w','MarkerSize',MSize);
    plot(min(x) - dP*MSize,1,'>','color',colorOrder(1,:),'MarkerFaceColor','w','MarkerSize',MSize);
    
    xlabel('Power');
    ylabel('Voltage');
end




if exportFig && pltPoster==0 && pltThesis==0
    export_fig(fig,FN);
    export_fig(fig,[FN,'.pdf'],'-dpdf');
    saveas(fig,FN,'meta')
    close
end

if exportFig && pltPoster
    export_fig(fig,[FN,'.pdf'],'-dpdf','-transparent');
    if pltPres==0
        a = 10
        close
   end

end

if exportFig && pltThesis
   export_fig(fig,[FN,'.pdf'],'-dpdf','-transparent');
    close   
end

if pltPoster && pltPres
   FN = 'C:\Users\Matt\Documents\DPhil\papers\pesgm19\pesgm19_presentation\figures\methodsPrs';
   export_fig(fig,FN);
   export_fig(fig,[FN,'.pdf'],'-dpdf','-transparent');
   saveas(fig,FN,'meta')
   close
end


