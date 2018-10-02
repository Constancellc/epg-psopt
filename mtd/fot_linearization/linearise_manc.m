% reproduce_nrel_full.m is a script which is designed to demonstrate and
% compare the accuracy of the nrel linearisations, both for the fixed point
% linearization (FPL) and first order taylor (FOT) methods. [at present,
% onlt the FPL method has been got up and running due to the relatively
% large matrix inversion required for the FOT method]

clear all; close all; clc;
%%
fig_loc = [pwd,'\figures\'];
% feeder_loc = '\13Bus_copy\IEEE13Nodeckt';
WD = 'C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\mtd\fot_linearization';
cd(WD);
addpath('lin_functions\');

fn = [WD,'\manchester_models\network_1\Feeder_1\master'];
feeder='n1f1';
% fn = [WD,'\manchester_models\network_2\Feeder_1\master'];
% feeder='n2f1';
% fn = [WD,'\manchester_models\network_3\Feeder_1\master'];
% feeder='n3f1';
% fn = [WD,'\manchester_models\network_4\Feeder_1\master'];
% feeder='n4f1';

fn_y = [fn,'_y'];
sn=[WD,'\',feeder];

lin_point=0.6;%kw
%% -----------------
% Run the DSS 
[~, DSSObj, DSSText] = DSSStartup;
DSSCircuit=DSSObj.ActiveCircuit; DSSSolution = DSSCircuit.Solution;
DSSText.command=['Compile (',fn,'.dss)'];
% [~] = set_cp_loads(DSSCircuit); % fix taps at their current positions

[ TC_No0,TR_name,TC_bus ] = find_tap_pos( DSSCircuit );

% H = calc_Hmat( DSSCircuit );
[Ybus,YZNodeOrder] = create_tapped_ybus( DSSObj,fn_y,feeder,TR_name,TC_No0 );

%% REPRODUCE the 'Delta Power Flow Eqns' (1)
DSSText.command=['Compile (',fn,'.dss)'];
[ BB00,SS00 ] = cpf_get_loads( DSSCircuit );
k00 = lin_point/real(SS00{1});
[~] = cpf_set_loads(DSSCircuit,BB00,SS00,k00);
DSSSolution.Solve;

% get the Y, D currents/powers
[B,V,I,S,D] = ld_vals( DSSCircuit );
[iD,sD,iY,sY] = calc_sYsD( YZNodeOrder,B,I,S,D );
[ BB0,SS0 ] = cpf_get_loads( DSSCircuit );

YNodeVarray = DSSCircuit.YNodeVarray';
YNodeV = YNodeVarray(1:2:end) + 1i*YNodeVarray(2:2:end);
% --------------------------
xhy0 = -1e3*[real(sY(4:end));imag(sY(4:end))];

V0 = YNodeV(1:3);
Vh = YNodeV(4:end);

Ybus_sp = sparse(Ybus);
tic
[ My,a ] = nrel_linearization_My( Ybus_sp,Vh,V0 );
toc
%% check
k = (-0.7:0.1:1.7);

v = zeros(numel(k),numel(YZNodeOrder));
v_l = zeros(numel(k),numel(YZNodeOrder) - 3);
v_l0 = zeros(numel(k),numel(YZNodeOrder) - 3);

ve = zeros(size(k));
ve0 = zeros(size(k));

DSSText.command=['Compile (',fn,')'];
% [~] = set_cp_loads(DSSCircuit);
[~] = set_taps(DSSCircuit.RegControls); % fix taps at their current positions

tic
for i = 1:numel(k)
    DSSText.command=['Compile (',fn,')'];
    % [~] = set_cp_loads(DSSCircuit); % fix taps at their current positions
    [~] = set_taps(DSSCircuit.RegControls); % fix taps at their current positions

    [~] = cpf_set_loads(DSSCircuit,BB0,SS0,k(i));
    DSSSolution.Solve;
    
    YNodeVarray = DSSCircuit.YNodeVarray';
    v(i,:) = YNodeVarray(1:2:end) + 1i*YNodeVarray(2:2:end);

    % a la risk day run_lin_model.m
    [B,V,I,S,D] = ld_vals( DSSCircuit );
    [iD,sD,~,sY] = calc_sYsD( YZNodeOrder,B,I,S,D );
    
    xhy = -1e3*[real(sY(4:end));imag(sY(4:end))];
    xhd = -1e3*[real(sD(4:end));imag(sD(4:end))];

    v_l(i,:) = My*xhy+ a;
    v_l0(i,:) = k(i)*My*xhy0 + a;

    ve(i) = norm([V0.',v_l(i,:)] - v(i,:))/norm(v(i,:));
    ve0(i) = norm([V0.',v_l0(i,:)] - v(i,:))/norm(v(i,:));
    
end
toc

plot(k,ve); hold on;
plot(k,ve0);
xlabel('k'); ylabel('|V - V_e|/|V|');

%%
save(sn,'My','a','Ybus_sp','lin_point','V0','xhy0')



