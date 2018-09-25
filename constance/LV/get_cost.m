%load('../../../Documents/lin_model.mat')
load('../../../Documents/lvtestcase_lin_CC.mat')

sY = csvread('sY.csv');

My = [zeros(3,5436);My];

My_r = zeros(2721,110);
My_i = zeros(2721,110);

x = zeros(110,1);
loc = [0];

c = 1;
for i=1:length(xhy)
    if xhy(i) ~= 0
        x(c) = xhy(i);
        loc(c) = i;
        My_r(:,c) = real(My(:,i));
        My_i(:,c) = imag(My(:,i));
        c = c+1;
    end
end
loc
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

an = angle(v);
vr = real(v);
vi = imag(v);

vt = sqrt(vr.*vr+vi.*vi);


M_ = diag(cos(an))*My_r+diag(sin(an))*My_i;
a_ = cos(an).*a_r+sin(an).*a_i;
v_ = M_*x+a_;

e = sum(abs(v_-vt)./vt)/2721
%{
vminA = 10000000;
vmaxA = 0;
lowestA = 0;
highestA = 0;

vminB = 10000000;
vmaxB = 0;
lowestB = 0;
highestB = 0;

vminC = 10000000;
vmaxC = 0;
lowestC = 0;
highestC = 0;

for i=4:length(v_)
    if round(an(i),1) == -0.5:
        if v_(i) < vminA
            vminA = v_(i);
            lowestA = i;
        end

        if v_(i) > vmax
            vmax = v_(i);
            highest = i;
        end
end


M_h = M_(highest,:);
a_h = a_(highest,:);
M_l = M_(lowest,:);
a_l = a_(lowest,:);

dlmwrite('Mh.csv',M_h,'delimiter',',','precision',16);
dlmwrite('Ml.csv',M_l,'delimiter',',','precision',16);
dlmwrite('ah.csv',a_h,'delimiter',',','precision',16);
dlmwrite('al.csv',a_l,'delimiter',',','precision',16);
%}


M_2 = M_(loc(1:55),:);
a_2 = a_(loc(1:55),:);
v_ = M_2*x+a_2;

my_model_losses = transpose(x)*P*x + transpose(q)*x + c

% but so far we've assumed a constant power factor, so can we reduce
% further? factor = x(56)/x(1)

xr = x(1:55);
alpha = x(56)/x(1);

Pr = P(1:55,1:55) + alpha*P(56:end,1:55) + alpha*P(1:55,56:end) + ...
    alpha^2*P(56:end,56:end);
qr = q(1:55) + alpha*q(56:end);
Mr = M_2(:,1:55)+alpha*M_2(:,56:end);

v = Mr*xr+a_2

const_pf_model_losses = transpose(xr)*Pr*xr + transpose(qr)*xr + c


dlmwrite('M_.csv',Mr,'delimiter',',','precision',16);
dlmwrite('a_.csv',a_2,'delimiter',',','precision',16);

dlmwrite('P_.csv',Pr,'delimiter',',','precision',16);
dlmwrite('q_.csv',qr,'delimiter',',','precision',16);
dlmwrite('c_.csv',c,'delimiter',',','precision',16);

%{
dlmwrite('P0.csv',P(1:55,1:55),'delimiter',',','precision',16);
dlmwrite('q0.csv',q(1:55),'delimiter',',','precision',16);
%}



