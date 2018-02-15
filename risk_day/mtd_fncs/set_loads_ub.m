function DSSC = set_loads_ub( DSSC,Sld )
% set_loads is a function that is designed to set ALL LOADS in the OpenDSS
% Circuit DSSC to Sld, where Sld is a specified in kVA.

kW = real(Sld);

PF = real(Sld).*sign(imag(Sld))./abs(Sld);
PF(isnan(PF)) = 1;



LDS = DSSC.loads;
ii = LDS.first;
i = 1;
while ii
    AB = DSSC.ActiveElement.Bus;
    phase = str2num(AB{1}(end));
    LDS.kW = kW(phase);
    LDS.PF = PF(phase);
    ii = LDS.next;
    i=i+1;
end

end