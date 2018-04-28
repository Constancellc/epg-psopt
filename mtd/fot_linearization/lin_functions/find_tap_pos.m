function [ TC_No,TR_name,TC_bus ] = find_tap_pos( DSSCircuit )

DSSCircuit.RegControls.First;
TC_No(1) = DSSCircuit.RegControls.TapNumber;
TR_name{1} = DSSCircuit.RegControls.Transformer;
DSSCircuit.SetActiveElement(['Transformer.',TR_name{end}]);
TC_bus{1} = DSSCircuit.ActiveElement.BusNames;

% measure tap numbers and bus locations
while DSSCircuit.RegControls.Next>0
    TC_No = [TC_No DSSCircuit.RegControls.TapNumber];
    TR_name = [TR_name DSSCircuit.RegControls.Transformer];
    DSSCircuit.SetActiveElement(['Transformer.',TR_name{end}]);
    TC_bus = [TC_bus {DSSCircuit.ActiveElement.BusNames}];
end

end

