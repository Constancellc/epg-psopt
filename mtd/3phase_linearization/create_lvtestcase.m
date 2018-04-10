% In this script, wecreate the LV network model and use it to calculate the
% power flow injection at the head of the feeder.
close all; clear all; clc;

WD = 'C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\mtd\3phase_linearization';
cd(WD);
addpath('lin_functions');

fn = [WD,'\LVTestCase_copy\Master'];
G = [1 -1 0;0 1 -1; -1 0 1]; %gamma matrix

GG.filename = fn;
GG.filename_v = [GG.filename,'_v'];
GG.filename_y = [GG.filename,'_y'];
%% ----------
% Run the DSS
[~, DSSObj, DSSText] = DSSStartup;
DSSCircuit=DSSObj.ActiveCircuit;
DSSSolution=DSSCircuit.Solution;
DSSText.command=['Compile (',GG.filename,')'];

% NB no taps in these models.
YZNodeOrder = DSSCircuit.YNodeOrder;
YNodeVarray = DSSCircuit.YNodeVarray';
YNodeV = YNodeVarray(1:2:end) + 1i*YNodeVarray(2:2:end);

[B,V,I,S,D] = ld_vals( DSSCircuit );
[iD,sD,iY,sY] = calc_sYsD( YZNodeOrder,B,I,S,D );
H = kron(eye(DSSCircuit.NumBuses - 1),G);

%
[ YNodeV0,Ybus,~,~,~,~ ] = linear_analysis_3ph( DSSObj,GG,0,[],'flat',[] );

YNodeI = Ybus*YNodeV;
YNodeS = YNodeV.*conj(YNodeI)/1e3;

I0 = Ybus*YNodeV0;

clc
bus_ma(YZNodeOrder,abs(YNodeV),angle(YNodeV)*180/pi,'Voltages (|V|,arg(V))');
bus_ma(YZNodeOrder,abs(YNodeI),angle(-YNodeI)*180/pi,'Current inj');
bus_ma(YZNodeOrder,real(YNodeS),imag(YNodeS),'Power (P,Q), kVAR');

%%
xhy = -1e3*[real(sY(4:end));imag(sY(4:end))];
xhd = -1e3*[real(sD(4:end));imag(sD(4:end))];
xh = [xhy;xhd];
v0 = YNodeV(1:3);

[ My,Md,a,Ky,Kd,b ] = nrel_linearization( xh,H,Ybus,YNodeV(4:end),v0 );

% define linear model:
vc = My*xhy + Md*xhd + a;
vm = Ky*xhy + Kd*xhd + b;

% Check the values of (14):
norm(vc - YNodeV(4:end))/norm(YNodeV(4:end))
% plot(abs(vc)); hold on;
plot(abs(YNodeV(4:end)./abs(vc)));

%%
Vlin = [v0;vc];
Slin = Vlin.*conj(Ybus*Vlin)/1e3;
%%
subplot(211)
plot((real(Slin)-real(YNodeS))); hold on; 
subplot(212)
plot(real(Slin)); hold on; 















