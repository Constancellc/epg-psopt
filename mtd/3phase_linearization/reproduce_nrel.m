% reproduce_nrel is a script that looks to reproduce the results of the
% paper "load-flow in multiphase distrbution networks: existence,
% uniqueness, and linear models", available at
% https://arxiv.org/abs/1702.03310

clear all; close all; clc;

fig_loc = [pwd,'\figures\'];
% feeder_loc = '\13Bus_copy\IEEE13Nodeckt';
WD = 'C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\mtd\3phase_linearization';
cd(WD);
addpath('lin_functions\');

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
YNodeS = YNodeV.*conj(YNodeI);

clc
bus_ma(YZNodeOrder,abs(YNodeV)/1e3,angle(YNodeV)*180/pi,'');
% bus_ma(YZNodeOrder,abs(YNodeI),angle(-YNodeI)*180/pi,'');
bus_ma(YZNodeOrder,real(YNodeS)/1e3,imag(YNodeS)/1e3,'');

%$


iYD = iD_iY(0,0,32.605,-122.1,4.8119,119.7);
abs(iYD)
angle(iYD)*180/pi;
%%
DSSText.command=['Compile (',GG.filename,'.dss)'];
[B,V,I,S,D] = ld_vals( DSSCircuit );

% iD = zeros(size(YZNodeOrder));
% sD = zeros(size(YZNodeOrder));
% for i = 1:numel(B)
%     idx{i,1} = find_node_idx(YZNodeOrder,B{i}(1:3));
%     if numel(B{i})>4 && D{i}==1
%         if strcmp(B{i}(end-3),'.') && strcmp(B{i}(end-1),'.')
%             ph1=str2num(B{i}(end-2));
%             ph2=str2num(B{i}(end));
%             if ph1==1 && ph2==2
%                 iD(idx{i,1}(1)) = I{i}(1);
%                 sD(idx{i,1}(1)) = S{i}(1) + S{i}(2);
%             elseif ph1==2 && ph2==3
%                 iD(idx{i,1}(2)) = I{i}(1);
%                 sD(idx{i,1}(2)) = S{i}(1) + S{i}(2);
%             elseif ph1==3 && ph2==1
%                 iD(idx{i,1}(3)) = I{i}(1);
%                 sD(idx{i,1}(3)) = S{i}(1) + S{i}(2);
%             end
%         end
%         
%     else
%         iD(idx{i,1}) = I{i}*exp(+1i*pi/6)/sqrt(3);
%         sD(idx{i,1}) = S{i};
%     end
%     
% end

[iD,sD,iY,sY] = calc_sYsD( YZNodeOrder,B,I,S,D );

bus_ma(YZNodeOrder,abs(iD),angle(iD)*180/pi,'');
% bus_ma(YZNodeOrder,abs(v)/1e3,angle(v)*180/pi,'');
% bus_ma(YZNodeOrder,real(sD),imag(sD),'');

%%
G = [1 -1 0;0 1 -1; -1 0 1];
HH = kron(eye(39),G);
% bus_ma(YZNodeOrder,abs(H*iD),angle(H*iD)*180/pi,'');

sD_p = diag((HH*YNodeV))*conj(iD);
sD_er = (real(sD_p)/1e3 - real(sD))./real(sD) + 1i*((imag(sD_p)/1e3 - imag(sD))./imag(sD));

bus_ma(YZNodeOrder,real(sD_er)*100,imag(sD_er)*100,'');

%% --------------------------
% FPL method (section IV.b), 37 node test feeder
% [ H ] = find_Hmat( DSSCircuit );
% we need to get the appropriate matrices (Y,Vh,H,xh)

Y = Ybus;

H = kron(eye(38),G);

% sY = zeros(size(sD));
xhy = -1e3*[real(sY(4:end));imag(sY(4:end))];
xhd = -1e3*[real(sD(4:end));imag(sD(4:end))];
xh = [xhy;xhd];

V0 = YNodeV(1:3);
Vh = YNodeV(4:end);
% vh = abs(YNodeV(4:end));

[ My,Md,a,Ky,Kd,b ] = nrel_linearization( xh,H,Ybus,Vh,V0 );
%
% V0 = Vh(1:3);
% V0=[1;exp(-1i*2*pi/3);exp(1i*2*pi/3)]*132*1e3;

% find derived values
% Nd = size(H,1);
% Ny = size(H,2);
% 
% xhy = xh(1:2*Ny);
% xhd = xh(2*Ny+1:end);
% 
% Yll = Y(4:end,4:end);
% Yl0 = Y(4:end,1:3);
% 
% w = -Yll\Yl0*V0;
% 
% % now create linearisation matrices
% My0 = inv(diag(conj(Vh))*Yll);
% Md0 = inv(Yll)*(H')*inv(diag(H*conj(Vh)));
% My = [ My0, -1i*My0 ]; %#ok<MINV>
% Md = [ Md0, -1i*Md0 ];
% 
% Ky = diag(vh)\real( diag(conj(Vh))*My );
% Kd  = diag(vh)\real( diag(conj(Vh))*Md );
% 
% a = w;
% b = vh - Ky*xhy - Kd*xhd;
 
% define linear model:
vc = My*xhy + Md*xhd + a;
vm = Ky*xhy + Kd*xhd + b;


%%
norm(vc - YNodeV(4:end))/norm(YNodeV(4:end))
% plot(abs(vc - YNodeV(4:end))./abs(YNodeV(4:end)) )


plot(abs(vc)); hold on;
plot(abs(YNodeV(4:end)));

%%
plot(abs(Md*xhd)*4800); hold on; 
plot(abs(a))



%%
bus_ma(YZNodeOrder(4:end),abs(vc)/1e3,angle(vc)*180/pi,'');


%%
plot(1e-3*abs(w(1:3:end))*sqrt(3)/4.8); hold on; 
plot(1e-3*abs(w(2:3:end))*sqrt(3)/4.8); 
plot(1e-3*abs(w(3:3:end))*sqrt(3)/4.8)

plot(1e-3*vm(1:3:end)*sqrt(3)/4.8); hold on; 
plot(1e-3*vm(2:3:end)*sqrt(3)/4.8); 
plot(1e-3*vm(3:3:end)*sqrt(3)/4.8)

plot(1e-3*abs(YNodeV(4:3:end))*sqrt(3)/4.8); hold on; 
plot(1e-3*abs(YNodeV(5:3:end))*sqrt(3)/4.8); 
plot(1e-3*abs(YNodeV(6:3:end))*sqrt(3)/4.8);









