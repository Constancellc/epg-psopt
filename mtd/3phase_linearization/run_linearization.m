close all; clear all; clc;
cd('C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\mtd\3phase_linearization')
addpath('lin_functions');
%% 13 BUS
NUT = '671';
pg_ssc = 10*linspace(-0.001,0.2,20);
qg_ssc = 10*linspace(-0.4,0.03,20);

% remain the same:
SRC = 'SOURCEBUS';
Asc3 = 200000; Asc1 = 2100; X1R1=4; X0R0=3; Zy_type = 'mva';
Cregs = {'Reg1','Reg2','Reg3'};
FF.feeder_loc = '\13Bus_copy\IEEE13Nodeckt_yy';
FF.filename_y = [pwd,FF.feeder_loc,'_y'];
FF.filename_v = [pwd,FF.feeder_loc,'_v'];
elmpwr_fnm = [pwd,'\13Bus_copy\IEEE13Nodeckt_EXP_ElemPowers.csv'];
feeder = '13bus';

% Run the DSS
[~, DSSObj, DSSText] = DSSStartup;

% First we need to find the nominal tap positions for the flat voltage profile
DSSText.command=['Compile (',pwd,FF.feeder_loc,'.dss)'];

DSSCircuit=DSSObj.ActiveCircuit;
DSSSolution=DSSCircuit.Solution;
DSSSolution.Solve;
DSSCircuit.Sample;

[ TC_No0,TR_name,TC_bus ] = find_tap_pos( DSSCircuit );

% With the taps known we calculate the linear A matrix and Ybus matrix
[ YNodeV0,Ybus0,~,n,Amat,~] = linear_analysis_3ph( DSSObj,FF,0,TR_name,TC_No0 );

% Now set up the full circuit with fixed taps
DSSText.command=['Compile (',pwd,FF.feeder_loc,'.dss)'];
DSSCircuit=DSSObj.ActiveCircuit;
if isempty(TR_name)==0
    if strcmp(feeder,'13bus')
        regname = 'RegControl.';
    elseif strcmp(feeder,'34bus')
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
YNodeVarray = DSSCircuit.YNodeVarray';
YNodeV = YNodeVarray(1:2:end) + 1i*YNodeVarray(2:2:end);
YNodeS = YNodeV.*conj(Ybus0*YNodeV);
YNodeS0 = YNodeV0.*conj(Ybus0*YNodeV0);
BB_n = [zeros(2*n,1);abs(YNodeV(1:3));angle(YNodeV(1:3));real(YNodeS(4:end));imag(YNodeS(4:end))];
BB_0 = [zeros(2*n,1);abs(YNodeV(1:3));angle(YNodeV(1:3));zeros(2*n-6,1)];

Xhat_n = Amat\BB_n;


% Withdraw unit bases
[Zbus,YZNodeOrder] = create_zbus(DSSCircuit);

DSSCircuit.SetActiveBus(NUT);
vbN = sqrt(3)*DSSCircuit.ActiveBus.kVBase; %NB this is LINE-NEUTRAL (not LL!)
DSSCircuit.SetActiveBus(SRC);
vbS = sqrt(3)*DSSCircuit.ActiveBus.kVbase;
nTr0 = vbN/vbS;
%
Vb = zeros(numel(YZNodeOrder),1);
for i = 1:numel(YZNodeOrder)
    DSSCircuit.SetActiveBus(YZNodeOrder{i});
    Vb(i) = DSSCircuit.ActiveBus.kVbase;
end

% PLOT the real and approximated values at nominal values
%plot(log(Xhat_n(1:n))); hold on; plot(log(abs(YNodeV)));

plot(VtBl,'o'); hold on; plot(1e-3*abs(YNodeV)./Vb,'*'); %plot(1e-3*Xhat_n(1:n)./Vb,'+');
%%
 hold on; plot(1e-3*abs(YNodeV)./Vb,'o'); grid on;
axis([0 45 0.95 1.1]);


%%
DSSCircuit.Transformers.First;
if strcmp(feeder,'34bus')
    sb = DSSCircuit.Transformers.kVA/10; %NB!!! for some reason they put this as 25000?
else
    sb = DSSCircuit.Transformers.kVA;
end

zbN = 1e3*(vbN^2)/sb; % in OHMS not kOhms.

% calculate source impedances 
[Z1_g, Z0_g] = calc_z0z1( Asc3,Asc1,X1R1,X0R0,vbN,Zy_type);
Zy_g = abs(Z0_g)*exp(1i*angle(Z1_g));
Z_ohm  = find_node_Z( YZNodeOrder,Zbus,SRC,NUT,nTr0,Zy_g,0 );
Z = Z_ohm/zbN;

Ssc = ((vbN^2)/abs(Z*zbN))*1e3; % kW

% create generator at given node:
DSSText.Command=['new generator.gen phases=3 bus1=',NUT,'.1.2.3 kw=0 pf=0'];
DSSText.Command='new monitor.genvi element=generator.gen terminal=1 mode=32 VIPolar=yes';
DSSText.Command='new monitor.genpq element=generator.gen terminal=1 mode=65 PPolar=no';
%%
[Pg, Qg] = meshgrid(Ssc*pg_ssc,Ssc*qg_ssc);
Pf = sign(Qg).*Pg./sqrt(Pg.^2 + Qg.^2);

Pl_mes_lin = zeros(size(Pg));
totlss = zeros(numel(Pg),2); totpwr = zeros(numel(Pg),2);
VmaxMes = zeros(numel(Pg),1); VminMes = zeros(numel(Pg),1);

DSSCircuit.Monitors.ResetAll;
tic
for i = 1:numel(Pg)
    DSSText.Command=strcat('edit generator.gen kw=',num2str(Pg(i)));
    DSSText.Command=strcat('edit generator.gen pf=',num2str(Pf(i)));
    DSSSolution.Solve;
    DSSCircuit.Sample;
    
    AllBusVmagPu = DSSCircuit.AllBusVmagPu; 
    VmaxMes(i) = max(AllBusVmagPu);
    VminMes(i) = min(AllBusVmagPu);
    totlss(i,:) = 1e-3*DSSObj.ActiveCircuit.Losses/sb; %pu. output seems to be in watts rather than kW
    totpwr(i,:) = DSSObj.ActiveCircuit.TotalPower/sb; %pu
end
toc
% withdraw results
DSSMon=DSSCircuit.Monitors; DSSMon.name='genvi';
DSSText.Command='show monitor genvi'; % this has to be in for some reason?
Vgen_lin = ExtractMonitorData(DSSMon,1,vbN*1e3/sqrt(3)); %phase 1
Vgen_lin2 = ExtractMonitorData(DSSMon,2,vbN*1e3/sqrt(3)); %phase 2
Vgen_lin3 = ExtractMonitorData(DSSMon,3,vbN*1e3/sqrt(3)); %phase 3
Vgen1=reshape(Vgen_lin,size(Pg));
Vgen2=reshape(Vgen_lin2,size(Pg));
Vgen3=reshape(Vgen_lin3,size(Pg));

% Vgen = max(cat(3,Vgen1,Vgen2,Vgen3),[],3); %assume voltage limit refers to a maximum phase LN voltage magnitude
Vgen = (Vgen1 + Vgen2 + Vgen3)/3;
VmaxMat = reshape(VmaxMes,size(Pg));
VminMat = reshape(VminMes,size(Pg));

DSSMon=DSSCircuit.Monitors; DSSMon.name='genpq';
DSSText.Command='show monitor genpq';
Pgen_lin = -ExtractMonitorData(DSSMon,1,1)/sb; %pu, +ve is generation
Qgen_lin = -ExtractMonitorData(DSSMon,2,1)/sb; %pu, +ve is generation

TotLss = reshape(totlss(:,1),size(Pg)) + 1i*reshape(totlss(:,2),size(Pg));% +ve implies losses
TotPwr = reshape(totpwr(:,1),size(Pg)) + 1i*reshape(totpwr(:,2),size(Pg)); %-ve implies load

Pgenmat=reshape(Pgen_lin,size(Pg));% - real(TotLd);
Qgenmat=reshape(Qgen_lin,size(Pg));% - imag(TotLd);
Sgenmat = Pgenmat+1i*Qgenmat;
%%
nut_idx = find_node_idx(YZNodeOrder,NUT);
nut_idx=nut_idx(nut_idx~=0);
Sg_mes_lin = sb*1e3*(Pgen_lin'+1i*Qgen_lin'); %watts

[ VgLin_n,Xnut_n ] = linear_solve( Amat,Sg_mes_lin,BB_n,nut_idx,Pg,vbN ); %volts
[ VgLin_0,Xnut_0 ] = linear_solve( Amat,Sg_mes_lin,BB_0,nut_idx,Pg,vbN ); %volts

fig = figure('Color','White');
subplot(231)
[cc,~]=contourf(Pgenmat,Qgenmat,Vgen);
clabel(cc); axis equal;
subplot(232)
[cc,~]=contourf(Pgenmat,Qgenmat,VgLin_0);
clabel(cc); axis equal;
subplot(233)
[cc,~]=contourf(Pgenmat,Qgenmat,VgLin_n);
clabel(cc); axis equal;

subplot(235)
[cc,~]=contourf(Pgenmat,Qgenmat,100*(VgLin_0-Vgen));
clabel(cc); axis equal; title('% error');
subplot(236)
[cc,~]=contourf(Pgenmat,Qgenmat,100*(VgLin_n-Vgen));
clabel(cc); axis equal; title('% error');

%%
fig = figure('Color','White');
subplot(331)
[cc,~]=contourf(Pgenmat,Qgenmat,Vgen1);
clabel(cc); axis equal; title('Meas (Phase 1)'); grid on;
subplot(332)
[cc,~]=contourf(Pgenmat,Qgenmat,Vgen2);
clabel(cc); axis equal; title('Meas (Phase 2)'); grid on;
subplot(333)
[cc,~]=contourf(Pgenmat,Qgenmat,Vgen3);
clabel(cc); axis equal; title('Meas (Phase 3)'); grid on;

subplot(334)
[cc,~]=contourf(Pgenmat,Qgenmat, reshape(Xnut_n(1,:),size(Pg)));
clabel(cc); axis equal; title('Lin approx.'); grid on;
subplot(335)
[cc,~]=contourf(Pgenmat,Qgenmat,reshape(Xnut_n(2,:),size(Pg)));
clabel(cc); axis equal; title('Lin approx.'); grid on;
subplot(336)
[cc,~]=contourf(Pgenmat,Qgenmat,reshape(Xnut_n(3,:),size(Pg)));
clabel(cc); axis equal; title('Lin approx.'); grid on;

subplot(337)
[cc,~]=contourf(Pgenmat,Qgenmat, (reshape(Xnut_n(1,:),size(Pg)) - Vgen1)*100);
clabel(cc); axis equal; title('% error'); grid on;
subplot(338)
[cc,~]=contourf(Pgenmat,Qgenmat, (reshape(Xnut_n(2,:),size(Pg)) - Vgen2)*100);
clabel(cc); axis equal; title('% error'); grid on;
subplot(339)
[cc,~]=contourf(Pgenmat,Qgenmat, (reshape(Xnut_n(3,:),size(Pg)) - Vgen3)*100);
clabel(cc); axis equal; title('% error'); grid on;





