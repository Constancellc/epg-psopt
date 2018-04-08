% lin_cpf is to be used to create a 'continuation power flow', using a
% parameter k, representing a scaling of the load.

close all; clear all; clc;

WD = 'C:\Users\Matt\Documents\MATLAB\epg-psopt\mtd\source_linearization';
cd(WD); addpath('mtd_fncs');

%%
% Run the nominal DSS
[~, DSSObj, DSSText] = DSSStartup;
DSSCircuit=DSSObj.ActiveCircuit;
DSSSolution=DSSCircuit.Solution;

fn = [WD,'\test'];



%%

sc3=10000;
sc1=10500;
x1r1=4;
x0r0=3;
vll=115;
[ Z1,Z0 ] = source_impedance( vll,sc3,sc1,x1r1,x0r0 )

[ Zmat,Ymat ] = symm_line( Z1,Z0 )

Z1a = 1.65 + 6.6*1i;
Z0a = 1.9 + 1i*5.7;
[ Zmat,Ymat ] = symm_line( Z1a,Z0a )


Y1a = [0.0764 - 1i*0.3056;0.0382 - 0.1527*1i];
Ymat = eye(3)*Y1a(1) + (ones(3)-eye(3))*Y1a(2)

Zmat = inv(Ymat)
%%
DSSText.command=['Compile (',fn,')'];

%invariants:
H = kron(eye(DSSCircuit.NumBuses - 1),G);
YZNodeOrder = DSSCircuit.YNodeOrder;

tic % ~25 seconds
[ YNodeV0,Ybus,~,~,~,~ ] = linear_analysis_3ph( DSSObj,GG,0,[],'flat',[] );
toc
% Yll = Ybus(4:end,4:end);
% Yl0 = Ybus(4:end,1:3);
% % % w = -Yll\Yl0*v0;
% % w = -Yll\Yl0*YNodeV0(1:3);
% % a = w;
DSSText.command=['Compile (',GG.filename,')'];

% %-------------2: hat(S) = 0.6 kW (0.95 PF)
% Phat = 0.6;
% DSSCircuit = set_loads(DSSCircuit,S0*Phat);
% DSSSolution.Solve;
% 
% YNodeVarray = DSSCircuit.YNodeVarray';
% YNodeV = YNodeVarray(1:2:end) + 1i*YNodeVarray(2:2:end);
% [B,V,I,S,D] = ld_vals( DSSCircuit );
% [~,sD,~,sY] = calc_sYsD( YZNodeOrder,B,I,S,D );
% 
% xhy = -1e3*[real(sY(4:end));imag(sY(4:end))];
% xhd = -1e3*[real(sD(4:end));imag(sD(4:end))];
% xh = [xhy;xhd];
% 
% v0 = YNodeV(1:3); vh = YNodeV(4:end);
% tic % ~40 seconds
% [ My2,~,a2,~,~,~ ] = nrel_linearization( xh,H,Ybus,[v0;vh] );
% toc
% % save([WD,'\datasets\lvtestcase_lin2.mat'],'My2','a2','v0','vh','sY','Ybus');
% save([WD,'\datasets\lvtestcase_lin_CC.mat'],'My2','a2','v0','vh','sY','Ybus');
% %







