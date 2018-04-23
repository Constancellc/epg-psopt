function Feeder = LoadFeeder(Archivo);
% Load data from an Execel file
Feeder.Options.Name = Archivo;
Feeder.Topology = xlsread(Archivo,'Topology');
Feeder.Configurations = xlsread(Archivo,'Configurations');
Feeder.Loads = xlsread(Archivo,'Loads');
Feeder.Regulators = xlsread(Archivo,'Regulators');
General = xlsread(Archivo,'General');
Feeder.Slack = General(1);
if length(General(:,1)>=10)
  Feeder.Vpu_slack_phase = General(5:7).*exp(j*General(8:10)*pi/180);
else
  Feeder.Vpu_slack_phase = exp(j*[0,-120,120]*pi/180);    
end
Feeder.Vnom = General(2);
SistInterna = General(3);
Feeder.Switches = xlsread(Archivo,'Switches');
Feeder.Graphic = xlsread(Archivo,'Graphic');
Feeder.NumN = 0;
Feeder.NumL = length(Feeder.Topology(:,1));
Feeder.NumC = length(Feeder.Loads(:,1));
Feeder.NumZ = 0;
Feeder.NumR = 0;
Feeder.NumS = 0;
if length(Feeder.Switches)>0
    Feeder.NumS = length(Feeder.Switches(:,1));
end
if length(Feeder.Regulators)>0
    Feeder.NumR = length(Feeder.Regulators(:,1));
end

% Cambiar el nombre de los nodos
if Feeder.NumS>0
  Feeder.Nodes_ID = unique([Feeder.Topology(:,1);Feeder.Topology(:,2);Feeder.Switches(:,1);Feeder.Switches(:,2)]);
else
  Feeder.Nodes_ID = unique([Feeder.Topology(:,1);Feeder.Topology(:,2)]);    
end

Tpl = Feeder.Topology;
for k = 1:Feeder.NumL
    n = Tpl(k,1);
    nn = find(Feeder.Nodes_ID==n);
    Feeder.Topology(k,1) = nn;
    
    n = Tpl(k,2);
    nn = find(Feeder.Nodes_ID==n);
    Feeder.Topology(k,2) = nn;        
end

Swtc = Feeder.Switches;
for k = 1:Feeder.NumS
    n = Swtc(k,1);
    nn = find(Feeder.Nodes_ID==n);
    Feeder.Switches(k,1) = nn;
    
    n = Swtc(k,2);
    nn = find(Feeder.Nodes_ID==n);
    Feeder.Switches(k,2) = nn;    
end

% incluir los switches cerrados en las lineas
if Feeder.NumS>0
sc = find(Feeder.Switches(:,3)==1);
ls = length(sc);
NM = zeros(ls,4);
NM(:,1) = Feeder.Switches(sc,1);
NM(:,2) = Feeder.Switches(sc,2);
NM(:,3) = 0;  % distancia cero
NM(:,4) = 1;  % configuracion 1 (no importa realmente porque la distancia es cero !
Feeder.Topology = [Feeder.Topology; NM];
Feeder.NumL = Feeder.NumL + ls;
end

NuevasCargas = zeros(Feeder.NumC,1);
for k = 1:Feeder.NumC
    n = Feeder.Loads(k,1);
    nn = find(Feeder.Nodes_ID==n);
    NuevasCargas(k) = nn;
end
Feeder.Loads(:,1) = NuevasCargas;
Feeder.Loads(:,4:9) = Feeder.Loads(:,4:9)*1000;  % pasar de kW a W
% Cambiar al sistema internacional si es el caso
if SistInterna==0
   Feeder.Topology(:,3) =  Feeder.Topology(:,3)/3280.8399;  % longitud pasa de ft a km
   Feeder.Configurations(:,3:20) = Feeder.Configurations(:,3:20)/1.609344;
end

% Impedancias de las lineas
Feeder.NumZ = length(Feeder.Configurations(:,1));
ZLIN = zeros(3,3,Feeder.NumZ);
BLIN = zeros(3,3,Feeder.NumZ);
for k = 1:Feeder.NumZ       
    ZLIN(1,1,k) = Feeder.Configurations(k,3)+j*Feeder.Configurations(k,9);
    ZLIN(1,2,k) = Feeder.Configurations(k,4)+j*Feeder.Configurations(k,10);
    ZLIN(2,1,k) = ZLIN(1,2,k);
    ZLIN(1,3,k) = Feeder.Configurations(k,5)+j*Feeder.Configurations(k,11);
    ZLIN(3,1,k) = ZLIN(1,3,k);
    ZLIN(2,2,k) = Feeder.Configurations(k,6)+j*Feeder.Configurations(k,12);
    ZLIN(2,3,k) = Feeder.Configurations(k,7)+j*Feeder.Configurations(k,13);
    ZLIN(3,2,k) = ZLIN(2,3,k);
    ZLIN(3,3,k) = Feeder.Configurations(k,8)+j*Feeder.Configurations(k,14);                
        
    BLIN(1,1,k) = Feeder.Configurations(k,15);
    BLIN(1,2,k) = Feeder.Configurations(k,16);
    BLIN(2,1,k) = BLIN(1,2,k);
    BLIN(1,3,k) = Feeder.Configurations(k,17);
    BLIN(3,1,k) = BLIN(1,3,k);
    BLIN(2,2,k) = Feeder.Configurations(k,18);
    BLIN(2,3,k) = Feeder.Configurations(k,19);
    BLIN(3,2,k) = BLIN(2,3,k);
    BLIN(3,3,k) = Feeder.Configurations(k,20);                        
end
Feeder.ZLIN = ZLIN;
Feeder.BLIN = 1/2*BLIN*1E-6;

% Importante:  Revisar por que es diferente:  No deberia
Feeder.NumN = length(Feeder.Nodes_ID);
if not(Feeder.NumN==Feeder.NumL+1)
    disp('Warning:  The system may be not radial because NumN != NumL+1');
end
for k = 1:Feeder.NumN
    n = Feeder.Graphic(k,1);
    nn = find(Feeder.Nodes_ID==n);
    NuevoGrafico(k,1) = nn;
end
Feeder.Graphic(:,1) = NuevoGrafico;
Feeder.Slack = find(Feeder.Nodes_ID==Feeder.Slack);

% cambiar los reguladores
if Feeder.NumR>0
   Feeder.Regulators(:,2:4) = Feeder.Regulators(:,2:4)*5/8/100 + 1;
end
Feeder.Options.DeltaLoadFlow = General(4);
%% Falta
% Revisar que las cargas esten conectadas a fases que si existan
% 