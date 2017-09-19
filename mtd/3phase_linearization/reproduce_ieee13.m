% This script was used simply to try and reproduce the IEEE13 bus results
% using OpenDSS. It turns out that you MUST make sure to specify the
% frequency base on your machine if you have been using the European LV
% Test Feeder! 

clear all 
close all
clc
cd('C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\mtd\3phase_linearization');
addpath('lin_functions\');


%% First withdraw nominal constants from Bolognani paper:
VbaseBl = 4160/sqrt(3);
SbaseBl = 5e6;
ZbaseBl = VbaseBl^2/SbaseBl;

fig_loc = [pwd,'\figures\'];

[YBl,vBl,tBl,pBl,qBl,nBl] = ieee13_mod();

YNodeOrderBl0={'650';'RG60';'632';'633';'634';'645';'646';'671';'680';'675';'692';'684';'611';'652'};
YNodeOrderBl = cell(numel(YNodeOrderBl0)*3,1);
for i = 1:numel(YNodeOrderBl0)
    for j = 1:3
        YNodeOrderBl{(i-1)*3 + j} = [YNodeOrderBl0{i},'.',num2str(j)];
    end
end

v_testfeederBl = [...
    1.0000  1.0000  1.0000  ;...
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

v_testfeederBl = reshape(v_testfeederBl.',3*nBl + 3,1);

t_testfeederBl = [...
    0.00    -120.00 120.00  ;...
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

t_testfeederBl = reshape(t_testfeederBl.',3*nBl + 3,1)/180*pi;


%%
feeder_loc = '\13Bus_copy\IEEE13Nodeckt';
% Run the DSS
[~, DSSObj, DSSText] = DSSStartup;

% First we need to find the nominal tap positions for the flat voltage profile
DSSText.command=['Compile (',pwd,feeder_loc,'.dss)'];

DSSCircuit=DSSObj.ActiveCircuit;
DSSSolution=DSSCircuit.Solution;
DSSSolution.Solve;

% Use to calculate nominal voltages:
YNodeVarray = DSSCircuit.YNodeVarray';
YNodeV = YNodeVarray(1:2:end) + 1i*YNodeVarray(2:2:end);
YZNodeOrder = DSSCircuit.YNodeOrder;

VtBl = NaN*zeros(numel(YZNodeOrder),1);
TtBl = NaN*zeros(numel(YZNodeOrder),1);
for i = 1:numel(YZNodeOrder)
    if ismember(YZNodeOrder{i},YNodeOrderBl)
        idx = find(strcmp(YNodeOrderBl,YZNodeOrder{i}));
        VtBl(i) = v_testfeederBl(idx);
        TtBl(i) = t_testfeederBl(idx);
    end
end


Vb = zeros(numel(YZNodeOrder),1);
for i = 1:numel(YZNodeOrder)
    DSSCircuit.SetActiveBus(YZNodeOrder{i});
    Vb(i) = DSSCircuit.ActiveBus.kVbase;
end


%% 

dugan_v = [1.0001 1.0001 1.0001 1.0001 1.0001 1.0001 1.0625 1.05 1.0687 1.0179 1.04 1.0151 0.99397...
    1.0217 0.99623 0.98957 1.0534 0.97915 1.0328 1.0157 1.031 1.0136 0.98957 1.0534 0.97915...
     0.98311 1.0557 0.97731 0.97514 0.98205 1.0107 1.0449 1.0035 1.021 1.0419 1.0177 0.98957...
     1.0534 0.97915 0.98763 0.97714];

dugan_order = {'SOURCEBUS.1','SOURCEBUS.2','SOURCEBUS.3','650.1','650.2','650.3','RG60.1'...
    'RG60.2','RG60.3','633.1','633.2','633.3','634.1','634.2','634.3','671.1',...
    '671.2','671.3','645.2','645.3','646.2','646.3','692.1','692.2','692.3',...
    '675.1','675.2','675.3','611.3','652.1','670.1','670.2','670.3','632.1',...
    '632.2','632.3','680.1','680.2','680.3','684.1','684.3'};

VtD = zeros(numel(YZNodeOrder),1);
TtD = zeros(numel(YZNodeOrder),1);
for i = 1:numel(YZNodeOrder)
   idx = find(strcmp(dugan_order,YZNodeOrder{i}));
    VtD(i) = dugan_v(idx);
    TtD(i) = dugan_v(idx);
end



%%
fig_name = [fig_loc,'reproduce_ieee13'];
fig = figure('Color','White');
plot(VtBl,'o'); hold on; 
plot(1e-3*abs(YNodeV)./Vb,'*'); %plot(1e-3*Xhat_n(1:n)./Vb,'+');
axis([0 45 0.95 1.1]);
xlabel('Buses'); ylabel('|V| (pu)'); grid on; legend('Documentation Pg. 8','OpenDSS');

export_fig(fig,fig_name);
export_fig(fig,[fig_name,'.pdf'],'-dpdf');

%%
% First we need to find the nominal tap positions for the flat voltage profile
feeder_loc_rd = 'C:\Users\chri3793\Documents\OpenDSS\ieee13_dugan\IEEE13Nodeckt';
DSSText.command=['Compile (',feeder_loc_rd,'.dss)'];

DSSCircuit=DSSObj.ActiveCircuit;
DSSSolution=DSSCircuit.Solution;
DSSSolution.Solve;

% Use to calculate nominal voltages:
YNodeVarray_rd = DSSCircuit.YNodeVarray';
YNodeV_rd = YNodeVarray_rd(1:2:end) + 1i*YNodeVarray_rd(2:2:end);
YZNodeOrder_rd = DSSCircuit.YNodeOrder;


fig_name = [fig_loc,'reproduce_ieee13_rd'];
fig = figure('Color','White');
plot(VtBl,'o'); hold on; 
plot(1e-3*abs(YNodeV_rd)./Vb,'*'); %plot(1e-3*Xhat_n(1:n)./Vb,'+');
plot(VtD,'+');
axis([0 45 0.95 1.1]);
xlabel('Buses'); ylabel('|V| (pu)'); grid on; legend('Documentation Pg. 8','OpenDSS (This machine)','OpenDSS (R.D. machine)');

% export_fig(fig,fig_name);
% export_fig(fig,[fig_name,'.pdf'],'-dpdf');