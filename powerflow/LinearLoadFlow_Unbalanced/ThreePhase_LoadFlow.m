function Res = ThreePhase_LoadFlow(Feeder);
% Unbalanced load flow in power distribution systems
if Feeder.Options.DeltaLoadFlow==0
   Res = ThreePhase_LoadFlow_linetoneutral(Feeder); 
else
   Feeder.Vslack_linetoline = [1 -1 0; 0 1 -1; -1 0 1]*Feeder.Vpu_slack_phase/sqrt(3);
   Res = ThreePhase_LoadFlow_delta(Feeder);    
end