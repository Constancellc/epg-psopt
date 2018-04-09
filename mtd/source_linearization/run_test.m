% lin_cpf is to be used to create a 'continuation power flow', using a
% parameter k, representing a scaling of the load.

close all; clear all; clc;

WD = 'C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\mtd\source_linearization';
cd(WD); addpath('mtd_fncs');

%%
Isc3=10000;
Isc1=10500;
vll=115;

% Isc3=3000;
% Isc1=5;
% vll=11;

x1r1=4;
x0r0=3;
[ Z1,Z0 ] = source_impedance( vll,Isc3,Isc1,x1r1,x0r0 )