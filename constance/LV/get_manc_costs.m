%load('../../../Documents/lin_model.mat')

load('../../../Documents/manc_lin_networks/n4f1.mat')

[r,c] = size(My);
My = [zeros(3,c);My];

% find number of households
nH = nnz(xhy0);

My_r = zeros(r+3,nH);
My_i = zeros(r+3,nH);

x = zeros(nH,1);

c = 1;
for i=1:length(xhy0)
    if xhy0(i) ~= 0
        x(c) = xhy0(i);
        My_r(:,c) = real(My(:,i));
        My_i(:,c) = imag(My(:,i));
        c = c+1;
    end
end

a_r = real(a);
a_i = imag(a);

a_r = [real(V0);a_r];
a_i = [imag(V0);a_i];

Y_r = real(Ybus_sp);
Y_i = imag(Ybus_sp);

P = transpose(My_r)*Y_r*My_r - transpose(My_r)*Y_i*My_i + ...
    transpose(My_i)*Y_i*My_r + transpose(My_i)*Y_r*My_i;

q = 2*transpose(My_r)*Y_r*a_r - transpose(My_i)*Y_i*a_r - ...
    transpose(My_r)*Y_i*a_i + transpose(My_i)*Y_i*a_r + ...
    transpose(My_r)*Y_i*a_i + 2*transpose(My_i)*Y_r*a_i;

c = transpose(a_r)*Y_r*a_r - transpose(a_r)*Y_i*a_i + ...
    transpose(a_i)*Y_i*a_r + transpose(a_i)*Y_r*a_i;

nH = round(nH/2)
alpha = x(nH+1)/x(1);

Pr = P(1:nH,1:nH) + alpha*P(nH+1:end,1:nH) + alpha*P(1:nH,nH+1:end) + ...
    alpha^2*P(nH+1:end,nH+1:end);
qr = q(1:nH) + alpha*q(nH+1:end);

dlmwrite('manc_models/P4.csv',Pr,'delimiter',',','precision',16);
dlmwrite('manc_models/q4.csv',qr,'delimiter',',','precision',16);
dlmwrite('manc_models/c4.csv',c,'delimiter',',','precision',16);