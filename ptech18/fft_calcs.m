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
subplot(221);
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

%% 05/02/19: Using Chebfun for CF calcs
Nt = 2e4;
Nt = 8e3;
% Nt = 4e3;
Nt = 2e3;
Nt = 2e2; % must be even, so that

th_k = 0.1;
% th_k = 0.3;
% th_k = 1.0;
th_k = 3.0;
% th_k = 10.0;
% th_k = 30.0;

Dt = 2;
dt = Dt/Nt;

x0f = 0.5; % 0.5 is the whole window.

t = (-Dt/2:dt:Dt/2);
dx = 2*pi/Dt;
Dx = 2*pi/dt;
x = (-Dx/2:dx:Dx/2);
x = x + (Dx/2);

x0 = x(round(x0f*length(x)));
% x0 = x(round(0.7*length(x))); % if using positive
[~,ix02] = min(abs(x-(2*x0)));
x02 = x(ix02);

% cf for gamma function
k = 2;
% th = 1.5/dt;
th = th_k/dt;

pdfG = @(x) (x>=0).*(x.^(k-1)).*exp(-x/th)/(gamma(k)*(th^k));
% pdfU = @(x) ((x>=0).*(x<=x02))/(2*x0); # This seems to match analytic results less well.
pdfU = @(x) ((x>0).*(x<x02) + 0.5*((x==0) + (x==x02)))/(2*x0);
pdfH = @(x) pdfU(x).*pdfG(x)*(2*x0);

% % CHECK WHAT THE FUNCTIONS LOOK LIKE
% plot(x,pdfG(x)); hold on;
% ylm = ylim;
% plot(x,pdfU(x));
% ylim(ylm);
% pdfG(x(end))

% % AGAIN
% plot(x,pdfG(x)); hold on;
% plot(x,pdfU(x),'x');
% plot(x,pdfH(x),'x')
%
% Ft = fftshift(fft(ifftshift(pdfU(x))))*dx; % to middle-zero is using pdf
% with negative part
% Gt = fftshift(fft(ifftshift(pdfG(x))))*dx;
% Ht = fftshift(fft(ifftshift(pdfH(x))))*dx;

Ft = fftshift(fft(pdfU(x)))*dx;
Gt = fftshift(fft(pdfG(x)))*dx;
Ht = fftshift(fft(pdfH(x)))*dx;

% % CHECK HOW to do fft ifft shifting:
% % Gx =  fftshift(ifft(ifftshift(Gt)))/pi;
% Gx =  ifft(ifftshift(Gt))/pi;
% plot(x,pdfG(x)); hold on;
% plot(x,Gx,'x');

% CFs for the functions.
f = @(t) sinc(x0*t/pi).*exp(-1i*t*x0);
g = @(t) (1 + 1i.*th.*t).^-k;

% CHECK THE FUNCTION matches the FT of the function we are looking at.
% subplot(121) % NB! ONLY Seems to approximate well for reasonable dt.
% plot(t,real(Gt)); hold on;
% plot(t,gr(t),'x');
% 
% subplot(122)
% plot(t,imag(Gt)); hold on;
% plot(t,gi(t),'x');

% % IT SEEMS That the frequency spectrum is slightly out (by ~1/kt)
kt = (length(t) + 1)/length(t);
% figure;
% pltng = plot(t,real(Ft)); hold on;
% plot(t,real(Ft),'*','Color',pltng.Color)
% plot(t,fr(t/kt),'x') % THIS works best.
% plot(t,fr(t),'o');
% plot(t,fr(t*kt),'+'); grid on;
%

% qwe = chebfunpref();
% wer = qwe;
% wer.splitPrefs.splitLength=2000;
% chebfunpref(wer);
% chebfunpref(qwe); % reset


fprintf('\n Create chebfuns \n')
tic
Gr = chebfun(gr,'splitting','on');
Gi = chebfun(gi,'splitting','on');
Gg = Gr + 1i*Gi;
Ggm = merge(Gg); % This seems to be the best version (most likely to go through)

% G = chebfun(g,'splitting','on');
% Gm = merge(G);

% % PLOT the different chebfuns:
% plot(G,'x-'); hold on;
% plot(Gm,'*-');
% plot(Gg,'o-'); hold on;
% plot(Ggm,'+-');
% Hc = conv(F,G);
% Hc = conv(F,Gm);
% Hc = conv(F,Gg);
% Hc = conv(F,Ggm);

F = chebfun(f);
% G = chebfun(g,'splitting','on');
% G0 = chebfun(g,[-1,-0.010,0.010,1]);
% G0 = chebfun(g,[-1,-0.010,0.010,1]);
% G0 = chebfun(g,[-1,-0.10,0.10,1]);
G0 = chebfun(g,[-1,1]);
toc
fprintf('\nStart convolution: \n')
tic
% Hc = conv(F,Ggm,'same');
% Hc2 = conv(F,G,'same');
% Hc2 = conv(F,Gm,'same');
Hc = conv(F,G0,'same');
toc
Hc = merge(Hc); % tidy up

% % CHECK THE CHEBFUN matches the function it is supposed to:
% plot(t,real(g(t))); hold on;
% plot(t,real(G(t)),'x');
% 
% plot(t,imag(g(t))); hold on;
% plot(t,imag(G(t)),'x');

% % TAKE THE INVERSE Fourier transformer of each of the functions
% Gxc = fftshift(ifft(ifftshift(G(t))))/pi;
% plot(x,pdfG(x)); hold on;
% plot(x,Gx,'x');
% plot(x,Gxc,'o');


% Fxc = fftshift(ifft(ifftshift(F(t))))/pi;
% plot(x,pdfU(x)); hold on;
% plot(x,Fxc,'x');


% % CHECK that the convolution by chebfun is accurate:
% H = conv(f(t),g(t),'same');
% plot(t,real(H)*dt); hold on;
% plot(t,real(Hc(t)),'x');
%
% % CHECK how chebfun convolution CF compares to the actual CF
% subplot(121)
% plot(t,real(Ht)); grid on; hold on;
% plot(t,real(Hc(t/kt))*x0/pi,'x');
% subplot(122)
% plot(t,imag(Ht)); grid on; hold on;
% plot(t,imag(Hc(t/kt))*x0/pi,'x')

% % CHECK if Hc is hermitian
% subplot(121)
% plot(t,abs(Hc(t)./conj(Hc(-t)))); % this seems to be accurate
% subplot(122)
% plot(t,angle(Hc(t)./conj(Hc(-t))));

% % CHECK if the convolved CF is matching what we want
% Hxcs = ifft(ifftshift(Hc(t/kt)),'symmetric')/pi;
% Hxcr = ifft(ifftshift(Hc(t/kt)))/pi;
% plot(x,pdfH(x)); hold on;
% plot(x,Hxcs*x0/pi); % seems to work a bit better than using ifft symmetric
% plot(x,real(Hxcr*x0/pi)); % seems to work a bit better than using ifft symmetric

% % Finally: resample Hc to see if we can double the frequency
% NB: extrapolation feels like a bad idea!
%
Tkk = [1.0 0.33 0.10 0.033 0.01 0.0033 0.001]; % seems to work well to ~1%
% Tkk = [1.0 0.33 0.10];

figure;
for i = 1:numel(Tkk)
    Hxcr = ifft(ifftshift(Hc(t*Tkk(i)/kt)))/pi;
    subplot(121)
    semilogx(x,Tkk(i)*real(Hxcr*x0/pi)); hold on;
    subplot(122)
    plot(x,Tkk(i)*real(Hxcr*x0/pi)); hold on;
%     sum(real(Hxcr))*dx/dt
end



