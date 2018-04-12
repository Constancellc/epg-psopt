function [Ybus,YNodeOrder] = create_tapped_ybus( DSSObj,fn_y,feeder,TR_name,TC_No0 )
%CREATE_TAPPED_YBUS Summary of this function goes here
%   Detailed explanation goes here


DSSText = DSSObj.Text;

% first get the ybus matrix:
DSSText.command=['Compile (',fn_y,')'];
DSSCircuit=DSSObj.ActiveCircuit;
DSSSolution=DSSCircuit.Solution;

if isempty(TR_name)==0
    if strcmp(feeder,'13bus')
        regname = 'RegControl.';
    elseif strcmp(feeder,'34bus') + strcmp(feeder,'37bus')
        regname = 'RegControl.c';
    end
    for i =1:numel(TR_name)
        DSSText.command=['edit ',regname,TR_name{i},' tapnum=',num2str(TC_No0(i))];
        DSSText.command=[regname,TR_name{i},'.maxtapchange=0']; % fix taps
    end
end

DSSSolution.Solve;
[Ybus_,YNodeOrder_] = create_ybus(DSSCircuit);
Ybus = Ybus_(4:end,4:end);
YNodeOrder = [YNodeOrder_(1:3);YNodeOrder_(7:end)];

end