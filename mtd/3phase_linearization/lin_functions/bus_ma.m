function [ YNodeOrder ] = bus_ma( YNodeOrder,XX,YY )
%BUS_MA Summary of this function goes here
%   Detailed explanation goes here

for i = 1:numel(YNodeOrder)
    fprintf('%s: (%.3d , %.3d) \n',YNodeOrder{i},XX(i),YY(i));
end


end

