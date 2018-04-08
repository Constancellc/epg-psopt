function [ Zmat,Ymat ] = symm_line( Z1,Z0 )
%SYMM_LINE Summary of this function goes here
%   Detailed explanation goes here

% Z1 = 6.6*1i + 1.65;
% Z0 = 5.7*1i + 1.9;

Zd = (2*Z1) - Z0;
Zo = Z0 - Z1;

Zmat = (1/3)*(eye(3)*Zd + (ones(3)-eye(3))*Zo);
Ymat = inv(Zmat);
end

