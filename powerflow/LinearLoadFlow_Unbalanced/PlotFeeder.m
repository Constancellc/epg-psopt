function  PlotFeeder(Feeder);
% Hacer el grafico
NumN = length(Feeder.Graphic(:,1));
NumL = length(Feeder.Topology(:,1));
r = size(Feeder.Regulators);
NumR = r(1);
plot(Feeder.Graphic(:,2),-Feeder.Graphic(:,3),'.b');
axis off
hold on
for k = 1:NumN
    text(Feeder.Graphic(k,2),-Feeder.Graphic(k,3),num2str(Feeder.Nodes_ID(Feeder.Graphic(k,1))));    
end
% Lineas
for k = 1:NumL
    N1 = Feeder.Topology(k,1);
    N2 = Feeder.Topology(k,2);
    c  = Feeder.Topology(k,4);    
    Zkm = abs(Feeder.ZLIN(:,:,c));
    H = sum(sign(sum(Zkm))');        
    P1 = find(Feeder.Graphic(:,1)==N1);
    P2 = find(Feeder.Graphic(:,1)==N2);
    S1 =  Feeder.Graphic([P1,P2],2);
    S2 = -Feeder.Graphic([P1,P2],3);      
    if H==1
       plot(S1,S2,'m')    
    end
    if H==2
       plot(S1,S2,'r')    
    end
    if H==3
       plot(S1,S2,'b')    
    end
    
end
% Reguladores
for k = 1:NumR
    R = Feeder.Regulators(k,1);    
    N1 = Feeder.Topology(R,1);
    N2 = Feeder.Topology(R,2);
    P1 = find(Feeder.Graphic(:,1)==N1);
    P2 = find(Feeder.Graphic(:,1)==N2);    
    SX = 0.5*(Feeder.Graphic(P1,2)+Feeder.Graphic(P2,2));
    SY = -0.5*(Feeder.Graphic(P1,3)+Feeder.Graphic(P2,3));
    plot(SX,SY,'ob','MarkerSize',12,'MarkerFaceColor',[0.7,0.7,0.9]);        
    text(SX-2,SY,'R');
end
% Switches
if length(Feeder.Switches)>0
for k = 1:length(Feeder.Switches(:,1))    
    N1 = Feeder.Switches(k,1);
    N2 = Feeder.Switches(k,2);
    P1 = find(Feeder.Graphic(:,1)==N1);
    P2 = find(Feeder.Graphic(:,1)==N2);    
    S1 = Feeder.Graphic([P1,P2],2);
    S2 = -Feeder.Graphic([P1,P2],3);
    SX = 0.5*(Feeder.Graphic(P1,2)+Feeder.Graphic(P2,2));
    SY = -0.5*(Feeder.Graphic(P1,3)+Feeder.Graphic(P2,3));            
    p = Feeder.Switches(k,3);
    if (p == 0) % switch abierto
        line(S1,S2,'Color',[.1 .8 .2]);            
        plot(SX,SY,'sg','MarkerFaceColor',[1,1,1]);        
    else
        line(S1,S2,'Color',[0 0 1]);
        plot(SX,SY,'sb');        
    end
end    
end
title (Feeder.Options.Name)
hold off

