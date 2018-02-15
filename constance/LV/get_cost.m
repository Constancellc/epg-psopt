%load('../../../Documents/lin_model.mat')
load('../../../Documents/lvtestcase_lin_CC.mat')

sY = csvread('sY.csv');

My = [zeros(3,5436);My];

My_r = zeros(2721,110);
My_i = zeros(2721,110);

x = zeros(110,1);

c = 1;
for i=1:length(xhy)
    if xhy(i) ~= 0
        x(c) = xhy(i);
        My_r(:,c) = real(My(:,i));
        My_i(:,c) = imag(My(:,i));
        c = c+1;
    end
end

a_r = real(a);
a_i = imag(a);

a_r = [real(v0);a_r];
a_i = [imag(v0);a_i];

Y_r = real(Ybus);
Y_i = imag(Ybus);

P = transpose(My_r)*Y_r*My_r - transpose(My_r)*Y_i*My_i + ...
    transpose(My_i)*Y_i*My_r + transpose(My_i)*Y_r*My_i;

q = 2*transpose(My_r)*Y_r*a_r - transpose(My_i)*Y_i*a_r - ...
    transpose(My_r)*Y_i*a_i + transpose(My_i)*Y_i*a_r + ...
    transpose(My_r)*Y_i*a_i + 2*transpose(My_i)*Y_r*a_i;

c = transpose(a_r)*Y_r*a_r - transpose(a_r)*Y_i*a_i + ...
    transpose(a_i)*Y_i*a_r + transpose(a_i)*Y_r*a_i;

% now test everything

%x = [-1e3*ones(55,1);-330*ones(55,1)];

%v = My*xhy+a_r+1i*a_i;
v = (My_r+1i*My_i)*x+a_r+1i*a_i;
S = v.*conj((Y_r+1i*Y_i)*v);
lin_model_losses = sum(S)

vr = real(v);
vi = imag(v);

my_model_losses = transpose(x)*P*x + transpose(q)*x + c

% but so far we've assumed a constant power factor, so can we reduce
% further? factor = x(56)/x(1)

xr = x(1:55);
alpha = x(56)/x(1);

Pr = P(1:55,1:55) + alpha*P(56:end,1:55) + alpha*P(1:55,56:end) + ...
    alpha^2*P(56:end,56:end);
qr = q(1:55) + alpha*q(56:end);

const_pf_model_losses = transpose(xr)*Pr*xr + transpose(qr)*xr + c

dlmwrite('P_.csv',Pr,'delimiter',',','precision',16);
dlmwrite('q_.csv',qr,'delimiter',',','precision',16);
dlmwrite('c_.csv',c,'delimiter',',','precision',16);

%{
dlmwrite('P0.csv',P(1:55,1:55),'delimiter',',','precision',16);
dlmwrite('q0.csv',q(1:55),'delimiter',',','precision',16);
%}



