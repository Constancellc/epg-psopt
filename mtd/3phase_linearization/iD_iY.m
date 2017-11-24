function iY = iD_iY( iM1,iP1,iM2,iP2,iM3,iP3 )


i12 = iM1*exp(1i*iP1*pi/180);
i23 = iM2*exp(1i*iP2*pi/180);
i31 = iM3*exp(1i*iP3*pi/180);

H = [1 -1 0;0 1 -1;-1 0 1];

iY = H'*[i12;i23;i31];



end

