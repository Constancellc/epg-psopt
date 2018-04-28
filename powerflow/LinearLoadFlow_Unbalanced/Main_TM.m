% Flujo de carga trifasico y comparacion con el lineal
clc
%clear all
% Tomar los datos

cd('C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\powerflow\LinearLoadFlow_Unbalanced');
%Feeder = LoadFeeder('FEEDER_IEEE_EULV_TM.xlsx');
Feeder = LoadFeeder('FEEDER_IEEE_EULV_TM2.xlsx');

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
%figure(2)
%bar(Error)
%title('Percentage Error')
%grid on

%%
figure(2)
plot(abs(Res.Vpu_phase))
xlabel('Bus')
ylabel('Voltage (pu)')
legend('a','b','c')

