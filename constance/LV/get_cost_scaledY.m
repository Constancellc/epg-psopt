%load('../../../Documents/lin_model.mat')
load('../../../Documents/lvtestcase_lin_CC.mat')

sY = csvread('sY.csv');
scale_factor = 100;

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

for sf = 1:10
Y_r = real(Ybus*scale_factor/(20*i));
Y_i = imag(Ybus*scale_factor/(20*i));

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

losses = transpose(x)*P*x + transpose(q)*x + c

end
