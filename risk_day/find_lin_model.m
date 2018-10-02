% lin_cpf is to be used to create a 'continuation power flow', using a
% parameter k, representing a scaling of the load.

close all; clear all; clc;

WD = 'C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\risk_day';
cd(WD); addpath('mtd_fncs');

fg_lc = 'C:\Users\chri3793\Documents\DPhil\risk_day\rd_ppt\figures\';   
fg_ps = [200 300 400 400];

set(0,'defaulttextinterpreter','latex');
set(0,'defaultaxesfontsize',14);
set(0,'defaulttextfontsize',14);

S0 = 1 + 1i*sqrt( (1 - 0.95^2)/(0.95^2) );
%%
% load('lvtestcase_lin.mat');

G = [1 -1 0;0 1 -1; -1 0 1]; %gamma matrix
fn = [WD,'\LVTestCase_copy\Master_z'];
% fn = [WD,'\LVTestCase_copy\Master'];

GG.filename = fn; 
GG.filename_v = [fn,'_v']; 
GG.filename_y = [fn,'_y'];
%%
% Run the nominal DSS
[~, DSSObj, DSSText] = DSSStartup;
DSSCircuit=DSSObj.ActiveCircuit;
DSSSolution=DSSCircuit.Solution;

DSSText.command=['Compile (',GG.filename,')'];
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

% %%
% %-------------1: hat(S) = 0.2 kW (0.95 PF)
% Phat = 0.2;
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
% [ My1,~,a1,~,~,~ ] = nrel_linearization( xh,H,Ybus,[v0;vh] );
% toc
% save([WD,'\datasets\lvtestcase_lin1.mat'],'My1','a1','v0','vh','sY','Ybus');

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
% %-------------3: hat(S) = 1.0 kW (0.95 PF)
% Phat = 1.0;
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
% [ My3,~,a3,~,~,~ ] = nrel_linearization( xh,H,Ybus,[v0;vh] );
% toc
% save([WD,'\datasets\lvtestcase_lin3.mat'],'My3','a3','v0','vh','sY','Ybus');
% 
% 
% %-------------4: hat(S) = 1.4 kW (0.95 PF)
% Phat = 1.4;
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
% [ My4,~,a4,~,~,~ ] = nrel_linearization( xh,H,Ybus,[v0;vh] );
% toc
% % save([WD,'\datasets\lvtestcase_lin4.mat'],'My4','a4','v0','vh','sY','Ybus');
%
%-------------CC: hat(S) = 0.6 kW (0.95 PF)
% Phat = 0.6;
% DSSCircuit = set_loads(DSSCircuit,S0*Phat);
% DSSSolution.Solve;

% YNodeVarray = DSSCircuit.YNodeVarray';
% YNodeV = YNodeVarray(1:2:end) + 1i*YNodeVarray(2:2:end);
% [B,V,I,S,D] = ld_vals( DSSCircuit );
% [~,sD,~,sY] = calc_sYsD( YZNodeOrder,B,I,S,D );

% xhy = -1e3*[real(sY(4:end));imag(sY(4:end))];
% xhd = -1e3*[real(sD(4:end));imag(sD(4:end))];
% xh = [xhy;xhd];

% v0 = YNodeV(1:3); vh = YNodeV(4:end);
% tic % ~40 seconds
% [ My,~,a,~,~,~ ] = nrel_linearization( xh,H,Ybus,[v0;vh] );
% toc
% save([WD,'\datasets\lvtestcase_lin_check.mat'],'My','a','v0','vh','sY','Ybus','xhy');Phat = 0.6;
% DSSCircuit = set_loads(DSSCircuit,S0*Phat);
% DSSSolution.Solve;

% YNodeVarray = DSSCircuit.YNodeVarray';
% YNodeV = YNodeVarray(1:2:end) + 1i*YNodeVarray(2:2:end);
% [B,V,I,S,D] = ld_vals( DSSCircuit );
% [~,sD,~,sY] = calc_sYsD( YZNodeOrder,B,I,S,D );

% xhy = -1e3*[real(sY(4:end));imag(sY(4:end))];
% xhd = -1e3*[real(sD(4:end));imag(sD(4:end))];
% xh = [xhy;xhd];

% v0 = YNodeV(1:3); vh = YNodeV(4:end);
% tic % ~40 seconds
% [ My,~,a,~,~,~ ] = nrel_linearization( xh,H,Ybus,[v0;vh] );
% toc
% save([WD,'\datasets\lvtestcase_lin_check.mat'],'My','a','v0','vh','sY','Ybus','xhy');









