function DSSC = set_load_prfl( DSSC,Sld )


kW = real(Sld);
PF = real(Sld).*sign(imag(Sld))./abs(Sld);
PF(isnan(PF)) = 1;

LDS = DSSC.loads;
ii = LDS.first;
i = 1;

while ii
    LDS.kW = kW(i);
    LDS.PF = PF(i);
    ii = LDS.next;
    i=i+1;
end



end