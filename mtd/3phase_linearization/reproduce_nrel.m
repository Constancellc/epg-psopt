% reproduce_nrel is a script that looks to reproduce the results of the
% paper "load-flow in multiphase distrbution networks: existence,
% uniqueness, and linear models", available at
% https://arxiv.org/abs/1702.03310
clear all; close all; clc;

fig_loc = [pwd,'\figures\'];
% feeder_loc = '\13Bus_copy\IEEE13Nodeckt';
WD = 'C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\mtd\3phase_linearization';
cd(WD);
addpath('lin_functions\');
%% -----------------
% Run the DSS 
[~, DSSObj, DSSText] = DSSStartup;
DSSCircuit=DSSObj.ActiveCircuit;

GG.filename = [WD,'\37Bus_copy\ieee37'];
GG.filename_v = [GG.filename,'_v'];
GG.filename_y = [GG.filename,'_y'];
GG.feeder='37bus';

% First we need to find the nominal tap positions for the flat voltage profile
DSSText.command=['Compile (',GG.filename,'.dss)'];
[ TC_No0,TR_name,TC_bus ] = find_tap_pos( DSSCircuit );
YZNodeOrder = DSSCircuit.YNodeOrder;
YNodeVarray = DSSCircuit.YNodeVarray';
YNodeV = YNodeVarray(1:2:end) + 1i*YNodeVarray(2:2:end);

[ YNodeV0,Ybus,~,~,~,~ ] = linear_analysis_3ph( DSSObj,GG,0,TR_name,'flat',TC_No0  );
YNodeI = Ybus*YNodeV;
YNodeS = YNodeV.*conj(YNodeI);

% Print some results:
clc
% bus_ma(YZNodeOrder,abs(YNodeV)/1e3,angle(YNodeV)*180/pi,'Voltage (|V| [kv], ang(V) [deg])');
% bus_ma(YZNodeOrder,abs(YNodeI),angle(-YNodeI)*180/pi,'Current(|I| [A], ang(I) [deg])');
% bus_ma(YZNodeOrder,real(YNodeS)/1e3,imag(YNodeS)/1e3,'Power (P [kW], Q [kVar])');

%% REPRODUCE the 'Delta Power Flow Eqns' (1)
DSSText.command=['Compile (',GG.filename,'.dss)'];
% get the Y, D currents/powers
[B,V,I,S,D] = ld_vals( DSSCircuit );
[iD,sD,iY,sY] = calc_sYsD( YZNodeOrder,B,I,S,D );

G = [1 -1 0;0 1 -1; -1 0 1];
HH = kron(eye(39),G);

sD_p = diag((HH*YNodeV))*conj(iD);
sD_er = (real(sD_p)/1e3 - real(sD))./real(sD) + 1i*((imag(sD_p)/1e3 - imag(sD))./imag(sD));
bus_ma(YZNodeOrder,real(sD_er)*100,imag(sD_er)*100,'');

% note that we get a small error out for the three phase load. This is
% presumed to be because we cannot specify iD from i exactly and so instead
% the currents have simply been assumed to shift by 30 degrees. If needed
% then the three individual 3-phase loads could be chosen to increase the
% accuracy.
%% --------------------------
% FPL method (section IV.b), 37 node test feeder
% [ H ] = find_Hmat( DSSCircuit );
% we need to get the appropriate matrices (Y,Vh,H,xh)

H = kron(eye(38),G);
xhy = -1e3*[real(sY(4:end));imag(sY(4:end))];
xhd = -1e3*[real(sD(4:end));imag(sD(4:end))];
xh = [xhy;xhd];

V0 = YNodeV(1:3);
Vh = YNodeV(4:end);

[ My,Md,a,Ky,Kd,b ] = nrel_linearization( xh,H,Ybus,Vh,V0 );

% define linear model:
vc = My*xhy + Md*xhd + a;
vm = Ky*xhy + Kd*xhd + b;

% Check the values of (14):
norm(vc - YNodeV(4:end))/norm(YNodeV(4:end))
plot(abs(vc)); hold on;
plot(abs(YNodeV(4:end)));










