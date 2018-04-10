% reproduce_nrel_full.m is a script which is designed to demonstrate and
% compare the accuracy of the nrel linearisations, both for the fixed point
% linearization (FPL) and first order taylor (FOT) methods. [at present,
% onlt the FPL method has been got up and running due to the relatively
% large matrix inversion required for the FOT method]

clear all; close all; clc;

fig_loc = [pwd,'\figures\'];
% feeder_loc = '\13Bus_copy\IEEE13Nodeckt';
WD = 'C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\mtd\3phase_linearization';
cd(WD);
addpath('lin_functions\');

% fn = [WD,'\37Bus_copy\ieee37'];
% feeder='37bus';

fn = [WD,'\34Bus_copy\ieee34Mod1_z'];
feeder='34bus';

fn_y = [fn,'_y'];
%% -----------------
% Run the DSS 
[~, DSSObj, DSSText] = DSSStartup;
DSSCircuit=DSSObj.ActiveCircuit;
% DSSText.command=['Compile (',fn,'.dss)'];
DSSText.command=['Compile (',fn,'.dss)'];
[ TC_No0,TR_name,TC_bus ] = find_tap_pos( DSSCircuit );

% G = [1 -1 0;0 1 -1; -1 0 1];
% H = kron(eye(DSSCircuit.NumBuses - 1),G);

H = calc_Hmat( DSSCircuit );
[Ybus,YZNodeOrder] = create_tapped_ybus( DSSObj,fn_y,feeder,TR_name,TC_No0 );

% First we need to find the nominal tap positions for the flat voltage profile
% DSSText.command=['Compile (',GG.filename,'.dss)'];
% [ TC_No0,TR_name,TC_bus ] = find_tap_pos( DSSCircuit );
% YNodeVarray = DSSCircuit.YNodeVarray';
% YNodeV = YNodeVarray(1:2:end) + 1i*YNodeVarray(2:2:end);

% [ YNodeV0,Ybus,~,~,~,~ ] = linear_analysis_3ph( DSSObj,GG,0,TR_name,'flat',TC_No0  );
% YNodeI = Ybus*YNodeV;
% YNodeS = YNodeV.*conj(YNodeI);
%% REPRODUCE the 'Delta Power Flow Eqns' (1)
% DSSText.command=['Compile (',fn,'.dss)'];
DSSText.command=['Compile (',fn,'.dss)'];
% get the Y, D currents/powers
[B,V,I,S,D] = ld_vals( DSSCircuit );
[iD,sD,iY,sY] = calc_sYsD( YZNodeOrder,B,I,S,D );

YNodeVarray = DSSCircuit.YNodeVarray';
YNodeV = YNodeVarray(1:2:end) + 1i*YNodeVarray(2:2:end);

% HH = kron(eye(39),G);

sD_p = diag((H*YNodeV(4:end)))*conj(iD(4:end));
sD_er = (real(sD_p)/1e3 - real(sD(4:end)))./real(sD(4:end)) + 1i*((imag(sD_p)/1e3 - imag(sD(4:end)))./imag(sD(4:end)));
% bus_ma(YZNodeOrder(4:end),real(sD_er)*100,imag(sD_er)*100,'');

%% --------------------------
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
legend('estimated','actual');
%%
bus_ma(YZNodeOrder(4:end),xhy(1:92),xhy(93:end),'Y');
bus_ma(YZNodeOrder(4:end),xhd(1:92),xhd(93:end),'Delta');

%%
plot(abs(vc)./abs(YNodeV(4:end)));








