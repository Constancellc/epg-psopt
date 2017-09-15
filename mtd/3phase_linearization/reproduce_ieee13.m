clear all 
close all
clc
cd('C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\mtd\3phase_linearization');
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
fig_name = [fig_loc,'reproduce_ieee13'];
fig = figure('Color','White');
plot(VtBl,'o'); hold on; plot(1e-3*abs(YNodeV)./Vb,'*'); %plot(1e-3*Xhat_n(1:n)./Vb,'+');
axis([0 45 0.95 1.1]);
xlabel('Buses'); ylabel('|V| (pu)'); grid on; legend('Documentation Pg. 8','OpenDSS');

export_fig(fig,fig_name);
export_fig(fig,[fig_name,'.pdf'],'-dpdf');


















