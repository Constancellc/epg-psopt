clear all;
% Nice resource:
% https://www.gaussianwaves.com/2014/07/how-to-plot-fft-using-matlab-fft-of-basic-signals-sine-and-cosine-waves/

% First example: rectangle to sinc. 
n = 2^16;
x = linspace(-20,20,n);

dx = x(2) - x(1);
dt = 1/(max(x) - min(x));
tmax = 0.5/dx;

t = linspace(-tmax,tmax,n);
N = n/2;

dN = 1;

pdfU = zeros(size(x)) + (abs(x)<0.5);
% pdfUfft = [pdfU(N+dN:end),pdfU(1:N-1+dN)];
pdfUfft = fftshift(pdfU);

% plot(pdfUfft)
CFU = fft(pdfU); % this does not give the nice clear 'real' result due to
% phase shift.
CFU = fft(pdfUfft);

figure;
plot(t,dx*real(fftshift(CFU))); hold on;
plot(t,dx*imag(fftshift(CFU)));
xlim([-10,10])
% plot(t,dx*real([CFU(N+1:end),CFU(1:N)])); hold on;
% plot(t,dx*imag([CFU(N+1:end),CFU(1:N)]));
% xlim([-10,10])

% figure;
% plot(t,dx*abs([CFU(N+1:end),CFU(1:N)])); hold on;
% plot(t,dx*angle([CFU(N+1:end),CFU(1:N)]));
% xlim([-10,10])

%%
% Gamma function, see: https://en.wikipedia.org/wiki/Gamma_distribution
k = 2.5; % shape
th = 0.1; % scale

tCF = -t*2*pi; % for some reason, -ve t, and scaled by 2*pi?

pdfG = zeros(size(x)) + (x>0).*(x.^(k-1)).*exp(-x/th)/((th^k)*gamma(k));
pdfGfft = [pdfG(N+1:end),pdfG(1:N)];
CFG = ((1 - th*1i*tCF).^(-k))/dx; % dx required to scale correctly
CFGa = fft(pdfG);
plot(x,pdfG); hold on;

pdfG_back = ifft(CFGa);
pdfG_back_an = ifft(CFG);

plot(x,pdfG_back)
plot(x,abs([pdfG_back_an(N+1:end),pdfG_back_an(1:N)]))
%%
subplot(221)
plot(t,abs(CFG));
subplot(222)
plot(t,unwrap(angle(CFG)) + 2*pi)
subplot(223)
plot(t,abs([CFGa(N+1:end),CFGa(1:N)])); hold on;
subplot(224)
plot(t,angle([CFGa(N+1:end),CFGa(1:N)]))

% plot(t,abs(CFGa))
% subplot(122)
% plot(t,angle(CFGa))

%% Difference of four gamma distributions with different parameters.
% gamma distrbution: https://en.wikipedia.org/wiki/Gamma_distribution
mults = [1,-1,2.5,-0.5];
k = [2,2,0.9,1.5]; % shape parameter
th = [0.15,0.65,1,0.8]; % scale parameter

intgt = 2;
intmax = 5;
dgnW = 1 - (intgt/intmax);

Nmc = 1e5; % number of monte carlo runs
% QWE = (randi([1,intmax],Nmc,1)>intgt); sum(QWE)/Nmc % checksum

g1 = gamrnd(k(1),th(1),Nmc,1).*(randi([1,intmax],Nmc,1)>intgt);
g2 = gamrnd(k(2),th(2),Nmc,1).*(randi([1,intmax],Nmc,1)>intgt);
g3 = gamrnd(k(3),th(3),Nmc,1).*(randi([1,intmax],Nmc,1)>intgt);
g4 = gamrnd(k(4),th(4),Nmc,1).*(randi([1,intmax],Nmc,1)>intgt);

gD = mults(1)*g1 + mults(2)*g2 + mults(3)*g3 + mults(4)*g4;

figure;
histogram(gD,3000,'Normalization','pdf'); hold on;
xlabel('x'); ylabel('p(x)');
xlim([-20,20]); grid on;

t= linspace(-100,100,1e3 + 1);
N = length(t);
cf = @(k,th,t,a,dgn) ((1 - th*1i*t*a).^(-k))*dgn + (1-dgn);

cf1 = cf(k(1),th(1),t,mults(1),dgnW);
cf2 = cf(k(2),th(2),t,mults(2),dgnW);
cf3 = cf(k(3),th(3),t,mults(3),dgnW);
cf4 = cf(k(4),th(4),t,mults(4),dgnW);

cf_tot = cf1.*cf2.*cf3.*cf4;

gDnew = abs(fftshift(ifft(cf_tot)))/(2*pi*dx);

dt = t(2) - t(1);
dx = 1/(max(t) - min(t));
x = -dx*[-N/2:1:(N/2)-1]*2*pi;

stairs(x,gDnew);
xlabel('x'); ylabel('p(x)');
xlim([-20,20]); grid on;
%%
subplot(121);
plot(t,abs(cf1)); hold on;
plot(t,abs(cf2));
subplot(122);
plot(t,angle(cf1)); hold on;
plot(t,angle(cf2));
