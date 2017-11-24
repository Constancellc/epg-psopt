% reproduce_nrel is a script that looks to reproduce the results of the
% paper "load-flow in multiphase distrbution networks: existence,
% uniqueness, and linear models", available at
% https://arxiv.org/abs/1702.03310

clear all; close all; clc;
addpath('lin_functions\');

fig_loc = [pwd,'\figures\'];
% feeder_loc = '\13Bus_copy\IEEE13Nodeckt';
WD = 'C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\mtd\3phase_linearization';
cd(WD);
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
YNodeS = YNodeV.*conj(YNodeI)/1e3;

clc
% bus_ma(YZNodeOrder,abs(YNodeV)/1e3,angle(YNodeV)*180/pi);
bus_ma(YZNodeOrder,abs(YNodeI),angle(-YNodeI)*180/pi);
% bus_ma(YZNodeOrder,real(YNodeS),imag(YNodeS));

%

iYD = iD_iY(0,0,32.605,-122.1,4.8119,119.7);
abs(iYD)
angle(iYD)*180/pi

%% --------------------------
% FPL method (section IV.b), 37 node test feeder
[ H ] = find_Hmat( DSSCircuit );
% we need to get the appropriate matrices (Y,Vh,H,xh)






% find derived values
Nd = size(H,1);
Ny = size(H,2);

xhy = xh(1:Ny);
xhd = xh(Ny+1:end);

V0 = Vh(1:3);

Yll = Y(4:end,4:end);
Yl0 = Y(4:end,1:3);

w = -Yll\Yl0*V0;

% now create linearisation matrices
My0 = inv(diag(conj(Vh))*Yll);
Md0 = Yll\(H')/diag(conj(H*Vh));
My = [ My0 , -1i*My0 ]; %#ok<MINV>
Md = [ Md0 , -1i*Md0 ];

Ky = diag(abs(vh))\real( diag(conj(vh))*My );
Kd = diag(abs(vh))\real( diag(conj(vh))*Md );

a = w;
b = abs(vh) - Ky*xhy - kd*xhd;

% define linear model:
vc = My*xy + Md*xd + a;
vm = Ky*xy + kd*xd + b;