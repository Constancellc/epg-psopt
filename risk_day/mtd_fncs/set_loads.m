function DSSC = set_loads( DSSC,Sld )
% set_loads is a function that is designed to set ALL LOADS in the OpenDSS
% Circuit DSSC to Sld, where Sld is a specified in kVA.

kW = real(Sld);
if Sld==0
    PF=1;
else
    PF = real(Sld)*sign(imag(Sld))/abs(Sld);
end

LDS = DSSC.loads;
ii = LDS.first;
i = 1;
while ii
    LDS.kW = kW;
    LDS.PF = PF;
    ii = LDS.next;
    i=i+1;
end

end