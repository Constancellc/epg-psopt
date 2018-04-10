% In this script, we attempt to find all of the current, power, and
% voltages of the YY YD DY DD transformer models, to ensure we fully know
% what is going on in OpenDSS.

close all; clear all; clc;

WD = 'C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\mtd\3phase_linearization';
cd(WD);
addpath('lin_functions');

fn_YY = [WD,'\4Bus_sets\4Bus-YY-Bal\4bus-YY-Bal'];
fn_DY = [WD,'\4Bus_sets\4Bus-DY-Bal\4bus-DY-Bal'];
fn_YD = [WD,'\4Bus_sets\4Bus-YD-Bal\4bus-YD-Bal'];
fn_GrdYD = [WD,'\4Bus_sets\4Bus-GrdYD-Bal\4bus-GrdYD-Bal'];
fn_OYOD_UB = [WD,'\4Bus_sets\4Bus-OYOD-UnBal\4bus-OYOD-UnBal'];

GG.filename = fn_OYOD_UB;
GG.filename_v = [GG.filename,'_v'];
GG.filename_y = [GG.filename,'_y'];
%% ----------
% Run the DSS
[~, DSSObj, DSSText] = DSSStartup;
DSSCircuit=DSSObj.ActiveCircuit;
DSSSolution=DSSCircuit.Solution;
DSSText.command=['Compile (',GG.filename,')'];

% [ TC_No0,TR_name,TC_bus ] = find_tap_pos( DSSCircuit );
YZNodeOrder = DSSCircuit.YNodeOrder;
YNodeVarray = DSSCircuit.YNodeVarray';
YNodeV = YNodeVarray(1:2:end) + 1i*YNodeVarray(2:2:end);

[ YNodeV0,Ybus,~,~,~,~ ] = linear_analysis_3ph( DSSObj,GG,0,[],'flat',[] );

YNodeI = Ybus*YNodeV;
YNodeS = YNodeV.*conj(YNodeI)/1e3;

I0 = Ybus*YNodeV0;

clc
bus_ma(YZNodeOrder,abs(YNodeV),angle(YNodeV)*180/pi,'');
bus_ma(YZNodeOrder,abs(YNodeI),angle(-YNodeI)*180/pi,'');
bus_ma(YZNodeOrder,real(YNodeS),imag(YNodeS),'');

%% to compare OYOD Unbalanced
% bus_ma(YZNodeOrder,real(-YNodeI),imag(-YNodeI),'');
iY = iD_iY( 302.43,-33.3,341.33,-157.7,542.38,84.9 );
abs(iY)
180*angle(iY)/pi

I12 =  252.88 - 1i*165.9; 
I23 = -315.88 - 1i*129.32;
I31 =  48.056 + 1i*540.25;
dI = [I12; I23; I31];

H = [1 -1 0;0 1 -1;-1 0 1];
H'*dI;







