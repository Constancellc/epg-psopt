function [ YNodeOrder ] = bus_ma( YNodeOrder,XX,YY )
%BUS_MA Summary of this function goes here
%   Detailed explanation goes here

fprintf('\n');
for i = 1:numel(YNodeOrder)
    fprintf('%s: (%.3f , %.3f) \n',YNodeOrder{i},XX(i),YY(i));
end
fprintf('\n');

end

