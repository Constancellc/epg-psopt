function Ybus = assemble_ybus( SystemY )

AA = SystemY(1:2:end);
BB = SystemY(2:2:end);
n = sqrt(numel(AA));

Ybus = reshape(AA+1i*BB,n,n);


end

