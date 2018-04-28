% This script is used to compare the 'true' LV test feeder network and the
% estimated feeder, using the calculated values of the source impedance to
% be inserted in a '_s' file.
% TWO things to measure: losses and voltages. Need to compare the actual
% with the nominal (ie huge source impedance) and the estimated.
%
% In order to create the file master_s, the 'run_test' script was used to
% find the appropriate (assumed) impedance of the soruce impedance line
% that is required.
%
% results discussed in WB 09/04/18, and in src_parameters.txt.
close all; clear all; clc;

WD = 'C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\mtd\source_linearization';
cd(WD); addpath('mtd_fncs');

fn = [WD,'\LVTestCase_copy\Master'];
%%
% Run the nominal DSS
[~, DSSObj, DSSText] = DSSStartup;
DSSCircuit=DSSObj.ActiveCircuit;
DSSSolution=DSSCircuit.Solution;
DSSText.command=['Compile (',fn,')'];

YZNodeOrder = DSSCircuit.YNodeOrder;
%% calculate impedance required
Isc3=3000; Isc1=5; vll=11; x1r1=4; x0r0=3;
[ Z1,Z0 ] = source_impedance( vll,Isc3,Isc1,x1r1,x0r0 )
S0 = 1 + 1i*sqrt( (1 - 0.95^2)/(0.95^2) );
k = (-0.75:0.05:1.75);
%% now run a bunch of loss calculations
v = zeros(numel(k),numel(YZNodeOrder));
sl = zeros(size(k));

DSSText.command=['Compile (',fn,')'];
for i = 1:numel(k)
    DSSCircuit = set_loads(DSSCircuit,S0*k(i));
    DSSSolution.Solve;

    YNodeVarray = DSSCircuit.YNodeVarray';
    v(i,:) = YNodeVarray(1:2:end) + 1i*YNodeVarray(2:2:end);
    
    DSSls = DSSCircuit.Losses;
    sl(i) = (DSSls(1) + 1i*DSSls(2))/1e3;
end

%%
v_s = zeros(numel(k),numel(YZNodeOrder)+3);
sl_s = zeros(size(k));
sl_s_line = zeros(size(k)); % USED to check if the error in losses that are
                            % measured are due to the extra line in the
                            % circuit that models the source impedance.

DSSText.command=['Compile (',fn,'_s)'];
for i = 1:numel(k)
    DSSCircuit = set_loads(DSSCircuit,S0*k(i));
    DSSSolution.Solve;

    YNodeVarray = DSSCircuit.YNodeVarray';
    v_s(i,:) = YNodeVarray(1:2:end) + 1i*YNodeVarray(2:2:end);
    
    DSSls = DSSCircuit.Losses;
    sl_s(i) = (DSSls(1) + 1i*DSSls(2))/1e3;
    
    DSSCircuit.SetActiveElement('Line.SourceLine');
    line_lss = DSSCircuit.ActiveElement.Losses;
    sl_s_line(i) = (line_lss(1) + 1i*line_lss(2))/1e3;
end

v_z = zeros(numel(k),numel(YZNodeOrder));
sl_z = zeros(size(k));

DSSText.command=['Compile (',fn,'_z)'];
for i = 1:numel(k)
    DSSCircuit = set_loads(DSSCircuit,S0*k(i));
    DSSSolution.Solve;

    YNodeVarray = DSSCircuit.YNodeVarray';
    v_z(i,:) = YNodeVarray(1:2:end) + 1i*YNodeVarray(2:2:end);
    
    DSSls = DSSCircuit.Losses;
    sl_z(i) = (DSSls(1) + 1i*DSSls(2))/1e3;
end
%%
cpfv = zeros(3,numel(k));
cpfl = zeros(3,numel(k));

for i = 1:numel(k)
    cpfv(1,i) = 100*norm(v(i,:) - v_z(i,:))/norm(v(i,:));
    cpfl(1,i) = 100*abs( sl(i) - sl_z(i)  )/abs( sl(i) );
    
    cpfv(2,i) = 100*norm(v(i,:) - v_s(i,4:end))/norm(v(i,:));
    cpfl(2,i) = 100*abs( sl(i) - sl_s(i)  )/abs( sl(i) );
    
    cpfv(3,i) = 100*norm(v(i,:) - v_s(i,4:end))/norm(v(i,:));
    cpfl(3,i) = 100*abs( sl(i) - (sl_s(i) - sl_s_line(i)) )/abs( sl(i) );
end

%%
figure('Color','White','Position',[200 100 450 650]);

subplot(211)
semilogy(k,cpfv')
grid on;
xlabel('k')
ylabel('|V - V''|_2/|V|_2 , %');
legend('Stiff','Stiff + Z_s','Stiff + Z_s + Comp.','Location','SouthEast');


subplot(212)
semilogy(k,cpfl')
grid on;
xlabel('k')
ylabel('|S_l - S_l''|/|S_l| , %');
legend('Stiff','Stiff + Z_s','Stiff + Z_s + Comp.','Location','SouthEast');

%%
figure('Color','White','Position',[200 100 800 650]);
subplot(222)
semilogy(k,cpfv')
grid on;
xlabel('k')
ylabel('|V - V''|_2/|V|_2 , %');

subplot(221)
plot(k,cpfv')
grid on;
xlabel('k')
ylabel('|V - V''|_2/|V|_2 , %');

subplot(224)
semilogy(k,cpfl')
grid on;
xlabel('k')
ylabel('|S_l - S_l''|/|S_l| , %');
% legend('Stiff','Stiff + Z_s','Stiff + Z_s + Comp.','Location','SouthEast');

subplot(223)
plot(k,cpfl')
grid on;
xlabel('k')
ylabel('|S_l - S_l''|/|S_l| , %');
legend('Stiff','Stiff + Z_s','Stiff + Z_s + Comp.','Location','East');

