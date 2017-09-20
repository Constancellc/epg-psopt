% A script that looks to reproduce the accuracy of the 'flat' solution seen
% in the paper 'Fast Power System Analysis via Implicit Linearization of
% the Power Flow Manifold' by Bolognani & Dorfler, 2015, using OpenDSS,
% with the inclusion of tap changers; transformers; using the Ybus creation
% capabilities of OpenDSS, to automate the procedure (and slightly as an
% exercise...).

clear all; close all; clc;

fig_loc = [pwd,'\figures\'];
addpath('lin_functions');
fig_pos = [100 200 520 540];

axis1 = [0 45 0.96 1.08];
axis2 = [0 45 -0.015 0.005];
%% First withdraw nominal constants from ETH paper:
VbETH = 4160/sqrt(3);
SbETH = 5e6;

[Yeth,~,~,pEth,qEth,nEth] = ieee13_mod();

v_testfeederEth = [...
    1.0625  1.0500  1.0687  ;...
    1.0210  1.0420  1.0174  ;...
    1.0180  1.0401  1.0148  ;...
    0.9940  1.0218  0.9960  ;...
    NaN     1.0329  1.0155  ;...
    NaN     1.0311  1.0134  ;...
    0.9900  1.0529  0.9778  ;...
    0.9900  1.0529  0.9777  ;...
    0.9835  1.0553  0.9758  ;...
    0.9900  1.0529  0.9778  ;...
    0.9881  NaN     0.9758  ;...
    NaN     NaN     0.9738  ;...
    0.9825  NaN     NaN     ];

v_testfeederEth = reshape(v_testfeederEth.',3*nEth,1);

t_testfeederEth = [...
    0.00    -120.00 120.00  ;...
    -2.49   -121.72 117.83  ;...
    -2.56   -121.77 117.82  ;...
    -3.23   -122.22 117.34  ;...
    NaN     -121.90 117.86  ;...
    NaN     -121.98 117.90  ;...
    -5.30   -122.34 116.02  ;...
    -5.31   -122.34 116.02  ;...
    -5.56   -122.52 116.03  ;...
    -5.30   -122.34 116.02  ;...
    -5.32   NaN     115.92  ;...
    NaN     NaN     115.78  ;...
    -5.25   NaN     NaN     ];

t_testfeederEth = reshape(t_testfeederEth.',3*nEth,1)/180*pi;

%% First, reproduce the original result.
a = exp(-1j*2*pi/3);
aaa = [1; a; a^2];

Veth = kron(ones(size(Yeth,1)/3,1),aaa);
Vbl_nf = kron(ones(size(Yeth,1)/3,1),aaa.*v_testfeederEth(1:3));

Aeth = calc_amat( Yeth,Veth );
Aeth_nf = calc_amat( Yeth,Vbl_nf );

BB_eth = [zeros(2*3*nEth,1);v_testfeederEth(1:3);t_testfeederEth(1:3);pEth(4:end);qEth(4:end)];

Xhat_eth = Aeth\BB_eth;
Xhat_ethnf = Aeth_nf\BB_eth;


% ------------ PLOT
figname = [fig_loc,'/ETH_sln'];
fig = figure('Color','White','Position',fig_pos);

subplot(211)
plot(v_testfeederEth,'o'); grid on; hold on; % cf eth_errors.fig in 'figures'
plot(Xhat_eth(1:3*nEth),'*');
plot(Xhat_ethnf(1:3*nEth),'x');
legend('True','1 pu','No ld','Location','SouthWest');
xlabel('Bus no.'); ylabel('Voltage (pu)'); title('ETH: problem solutions');
axis(axis1);

subplot(212);
plot(0,0); grid on; hold on;
pl1 = plot(v_testfeederEth - Xhat_eth(1:3*nEth),'*'); % cf eth_errors.fig in 'figures'
pl2 = plot(v_testfeederEth - Xhat_ethnf(1:3*nEth),'x');
legend([pl1,pl2],'Error, 1 pu','Error, no ld','Location','SouthWest');
xlabel('Bus no.'); ylabel('dV (pu)'); title('Solution error');
axis(axis2);
% export_fig(fig,figname);
% export_fig(fig,[figname,'.pdf'],'-dpdf');

%% Run the no-transformer model
[~, DSSObj, DSSText] = DSSStartup;
FF.filename = '\13Bus_copy\IEEE13Node_notr';
FF.filename_y = [pwd,FF.filename,'_y'];
FF.filename_v = [pwd,FF.filename,'_v'];

[ ~,Ybus,~,n,AmatV,~] = linear_analysis_3ph( DSSObj,FF,0,[],'flat' );
[ ~,~,~,~   ,AmatU,~] = linear_analysis_3ph( DSSObj,FF,0,[],'nold' );

DSSText.command=['Compile (',pwd,FF.filename,'.dss)'];
DSSCircuit=DSSObj.ActiveCircuit; DSSSolution=DSSCircuit.Solution; DSSSolution.Solve;

YNodeVarray = DSSCircuit.YNodeVarray';
YNodeV = YNodeVarray(1:2:end) + 1i*YNodeVarray(2:2:end);
YNodeS = YNodeV.*conj(Ybus*YNodeV);
YZNodeOrder = DSSCircuit.YNodeOrder;

BB_eth = [zeros(2*n,1);abs(YNodeV(1:3));angle(YNodeV(1:3));...
                    real(YNodeS(4:end));imag(YNodeS(4:end))];
Vb = zeros(numel(YZNodeOrder),1);
for i = 1:numel(YZNodeOrder)
    DSSCircuit.SetActiveBus(YZNodeOrder{i});
    Vb(i) = DSSCircuit.ActiveBus.kVbase;
end

BB_n = [zeros(2*n,1);abs(YNodeV(1:3));angle(YNodeV(1:3));-real(YNodeS(4:end));-imag(YNodeS(4:end))];

Xhat_nV = AmatV\BB_n;
Xhat_nU = AmatU\BB_n;


% ------------ PLOT
figname = [fig_loc,'/Opendss_sln'];
fig = figure('Color','White','Position',fig_pos);

subplot(211)
plot(1e-3*abs(YNodeV)./Vb,'o'); grid on; hold on;
plot(1e-3*Xhat_nV(1:n)./Vb,'*');
plot(1e-3*Xhat_nU(1:n)./Vb,'+');
legend('True','1 pu','No ld','Location','SouthWest');
xlabel('Bus no.'); ylabel('Voltage (pu)'); title('Opendss: problem solutions');
axis(axis1);

subplot(212);
plot(0,0); grid on; hold on;
pl1 = plot(1e-3*(abs(YNodeV) - Xhat_nV(1:n))./Vb,'*');
pl2 = plot(1e-3*(abs(YNodeV) - Xhat_nU(1:n))./Vb,'+');
legend([pl1,pl2],'Error, 1 pu','Error, no ld','Location','SouthWest');
xlabel('Bus no.'); ylabel('dV (pu)'); title('Solution error');
axis(axis2);

% export_fig(fig,figname);
% export_fig(fig,[figname,'.pdf'],'-dpdf');

%% Full transformer model
FF.filename = '\13Bus_copy\IEEE13Nodeckt_yy';
FF.filename_y = [pwd,FF.filename,'_y'];
FF.filename_v = [pwd,FF.filename,'_v'];
FF.feeder = '13bus';

DSSText.command=['Compile (',pwd,FF.filename,'.dss)'];
DSSCircuit=DSSObj.ActiveCircuit;
DSSSolution=DSSCircuit.Solution; DSSSolution.Solve; DSSCircuit.Sample;

[ TC_No0,TR_name,TC_bus ] = find_tap_pos( DSSCircuit );

% With the taps known we calculate the linear A matrix and Ybus matrix
[ YNodeV0,Ybus0,~,n,AmatV,~] = linear_analysis_3ph( DSSObj,FF,0,TR_name,'nold',TC_No0 );
[ ~,~,~,~,AmatW,~] = linear_analysis_3ph( DSSObj,FF,0,TR_name,'whtd',TC_No0 );
[ ~,~,~,~,AmatU,~] = linear_analysis_3ph( DSSObj,FF,0,TR_name,'flat',TC_No0 );

% Now set up the full circuit with fixed taps
DSSText.command=['Compile (',pwd,FF.filename,'.dss)'];
DSSCircuit=DSSObj.ActiveCircuit;
if isempty(TR_name)==0
    if strcmp(FF.feeder,'13bus')
        regname = 'RegControl.';
    elseif strcmp(FF.feeder,'34bus')
        regname = 'RegControl.c';
    end
    for i =1:numel(TR_name)
        DSSText.command=['edit ,',regname,TR_name{i},' tapnum=',num2str(TC_No0(i))];
        DSSText.command=[regname,TR_name{i},'.maxtapchange=0']; % fix taps
    end
end
DSSSolution=DSSCircuit.Solution;
DSSSolution.Solve;

% Use to calculate nominal voltages:
YZNodeOrder = DSSCircuit.YNodeOrder;

Vb = zeros(numel(YZNodeOrder),1);
for i = 1:numel(YZNodeOrder)
    DSSCircuit.SetActiveBus(YZNodeOrder{i});
    Vb(i) = DSSCircuit.ActiveBus.kVbase;
end

% Use to calculate nominal voltages:
YNodeVarray = DSSCircuit.YNodeVarray';
YNodeV = YNodeVarray(1:2:end) + 1i*YNodeVarray(2:2:end);
YNodeS = YNodeV.*conj(Ybus0*YNodeV);

BB_n = [zeros(2*n,1);abs(YNodeV(1:3));angle(YNodeV(1:3));-real(YNodeS(4:end));-imag(YNodeS(4:end))];

Xhat_nU = AmatU\BB_n;
Xhat_nV = AmatV\BB_n;
Xhat_nW = AmatW\BB_n;

% ------------ PLOT
figname = [fig_loc,'/Tr_sln'];
fig = figure('Color','White','Position',fig_pos);

subplot(211);
plot(1e-3*abs(YNodeV)./Vb,'o');hold on; grid on; 
plot(1e-3*Xhat_nW(1:n)./Vb,'*');
plot(1e-3*Xhat_nV(1:n)./Vb,'+');
legend('True','Weighted','No ld','Location','East');
xlabel('Bus no.'); ylabel('Voltage (pu)'); title('Opendss w/ Transformer: problem solutions');
axis(axis1);

subplot(212);
plot(0,0);hold on; grid on;
pl1=plot(1e-3*(abs(YNodeV)-Xhat_nW(1:n))./Vb,'*');
pl2=plot(1e-3*(abs(YNodeV)-Xhat_nV(1:n))./Vb,'+');
legend([pl1,pl2],'Error, weighted','Error, no ld','Location','SouthWest');
xlabel('Bus no.'); ylabel('dV (pu)'); title('Solution error');
axis(axis2);

% export_fig(fig,figname);
% export_fig(fig,[figname,'.pdf'],'-dpdf')
%% Finally plot the 'wrong' solution with a flat voltage profile.

% ------------ PLOT
figname = [fig_loc,'/Tr_fault'];
fig = figure('Color','White','Position',fig_pos);

subplot(211);
plot(1e-3*abs(YNodeV)./Vb,'o');hold on; grid on; 
plot(1e-3*Xhat_nU(1:n)./Vb,'*');
plot(1e-3*Xhat_nV(1:n)./Vb,'+');
legend('True','1 pu','No ld');%,'Location','East'
xlabel('Bus no.'); ylabel('Voltage (pu)'); title('Opendss w/ Transformer: problem solutions');

subplot(212);
plot(0,0);hold on; grid on;
pl1=plot(1e-3*(abs(YNodeV)-Xhat_nU(1:n))./Vb,'*');
pl2=plot(1e-3*(abs(YNodeV)-Xhat_nV(1:n))./Vb,'+');
legend([pl1,pl2],'Error, 1 pu','Error, no ld','Location','West');
xlabel('Bus no.'); ylabel('dV (pu)'); title('Solution error');

% export_fig(fig,figname);
% export_fig(fig,[figname,'.pdf'],'-dpdf')




