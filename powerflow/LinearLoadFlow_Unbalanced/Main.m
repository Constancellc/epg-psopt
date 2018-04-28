% Flujo de carga trifasico y comparacion con el lineal
clc
%clear all
% Tomar los datos
Feeder = LoadFeeder('FEEDER IEEE123_SINREGULADOR.xlsx');
% Dibujar
figure(1)
PlotFeeder(Feeder);
% Flujo de carga trifasico
Res = ThreePhase_LoadFlow(Feeder);
ShowResults(Res,Feeder);
% Ejemplo lineal
disp('Linear Load Flow');
ResL = Linear_Load_Flow_Unbalanced(Feeder);
ShowResults(ResL,Feeder);

if Feeder.Options.DeltaLoadFlow
   Error = abs(Res.Vpu_line-ResL.Vpu_line);     
else
   Error = abs(Res.Vpu_phase-ResL.Vpu_phase);
end
figure(2)
bar(Error)
title('Percentage Error')
grid on


