% run_linear_methods is a script that is used to test the function
% 
clear all; close all;
addpath('bolognani_fncs');

%%
c_name = 'case30';

[ SLN ] = linear_methods( c_name, 'nominal',0  );
[ SLN_flat ] = linear_methods( c_name, 'flat',0 );
[ SLN_LC ] = linear_methods( c_name, 'LC',0 );
[ SLN_DC ] = linear_methods( c_name, 'DC',0 );
[ SLN_LDF ] = linear_methods( c_name, 'LinDistFlow',0 );

n = SLN.n;

fig = figure('Color','White','Position',[100 100 850 650]);

subplot(211)
plot(1:n, SLN.VM, 'o'); hold on;
plot(SLN_flat.approxVM, '*'); hold on;
plot(SLN_LC.approxVM, '*'); hold on;
plot(SLN_DC.approxVM, '*'); hold on;
plot(SLN_LDF.approxVM, '*'); hold on;
ylabel('magnitudes [p.u.]')
xlim([0 n]); xlabel('Bus no.');
grid on;

subplot(212)
plot(1:n, SLN.VA, 'o'); hold on;
plot(SLN_flat.approxVA, '*'); hold on;
plot(SLN_LC.approxVA, '*'); hold on;
plot(SLN_DC.approxVM, '*'); hold on;
plot(SLN_LDF.approxVA, '*'); hold on;
ylabel('angles [deg]')
xlim([0 n]); xlabel('Bus no.');
grid on;

legend('AC Solution','Nominal','Lin. Coupled','DC','LinDistFlow','Location','SouthWest');











