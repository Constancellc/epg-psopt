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
a = a_r+1i*a_i;

Y_r = real(Ybus);
Y_i = imag(Ybus);

v = (My_r+1i*My_i)*x+a_r+1i*a_i;
M = My_r+1i*My_i;
Line2 = csvread('LVTest_EXP_YPRIM.csv',15,0,[15,0,20,11]);
Line2_ = Line2(:,1:2:end)+1i*Line2(:,2:2:end);

Line110 = csvread('LVTest_EXP_YPRIM.csv',771,0,[771,0,776,11]);
Line110_ = Line110(1:3,1:2:end)+1i*Line110(1:3,2:2:end);

Line296 = csvread('LVTest_EXP_YPRIM.csv',2073,0,[2073,0,2078,11]);
Line296_ = Line296(1:3,1:2:end)+1i*Line296(1:3,2:2:end);

v108 = v(3*108+1:3*108+3);
v111 = v(3*111+1:3*111+3);
v297 = v(3*297+1:3*297+3);
v288 = v(3*288+1:3*288+3);

va = v(7:9);
vb = v(10:12);
A = [Line110_*[M(3*108+1:3*108+3,:);M(3*111+1:3*111+3,:)];
    Line296_*[M(3*288+1:3*288+3,:);M(3*297+1:3*297+3,:)]];
A = A(:,1:55)+0.33*A(:,56:end);
Ar = real(A);
Ai = imag(A);

b = [Line110_*[a(3*108+1:3*108+3,:);a(3*111+1:3*111+3,:)];
    Line296_*[a(3*288+1:3*288+3,:);a(3*297+1:3*297+3,:)]];
br = real(b);
bi = imag(b);

dlmwrite('Ar.csv',Ar,'delimiter',',','precision',16);
dlmwrite('Ai.csv',Ai,'delimiter',',','precision',16);
dlmwrite('br.csv',br,'delimiter',',','precision',16);
dlmwrite('bi.csv',bi,'delimiter',',','precision',16);
%Line110_*[v108;v111]
%Line296_*[v288;v297]

% I want to make a matrix which will track my current against time at two
% lines - line 110 and line 296
