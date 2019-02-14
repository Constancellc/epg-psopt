% fftSum is a script to implement the experimental work from fft_calcs for
% using chebfun convolution to calculate total PDFs.

Nt = 8e3; % This is about as high as we go.
% Nt = 2e3;
% Nt = 2e2;

th_k = 0.3;
th_k = 1.0; 
th_k = 3.0;
% th_k = 30.0;

Dt = 2;
dt = Dt/Nt;

t = (-Dt/2:dt:Dt/2);
dx = 2*pi/Dt;
Dx = 2*pi/dt;
x = (-Dx/2:dx:Dx/2);
x = x + (Dx/2);

% cf for gamma function
k = 2;
th = th_k/dt;

x0 = x(round(0.5*length(x)));
%%
% CFs for the functions.
f = @(t) sinc(x0*t/pi).*exp(-1i*t*x0);
g = @(t) (1 + 1i.*th.*t).^-k;

% % IT SEEMS That the frequency spectrum is slightly out (by ~1/kt)
kt = (length(t) + 1)/length(t);

fprintf('\n Create chebfuns \n') % =======
tic
F = chebfun(f);
G = chebfun(g);
if length(G)>chebfunpref().techPrefs.maxLength
    G = chebfun(g,[-1,-0.1,0.1,1]);
    if length(G)>chebfunpref().techPrefs.maxLength
        G = chebfun(g,[-1,-0.01,0.01,1]);
    end
end

toc
fprintf('\nStart convolution: \n'); tic; % =======
Hc = conv(F,G,'same');
Hc = merge(Hc); % tidy up
toc

Tkk = [1.0 0.33 0.10 0.033 0.01 0.0033 0.001]; % seems to work well to ~1%
fprintf('\nStart Testing: \n'); tic;
figure;
for i = 1:numel(Tkk)
    Hxcr = ifft(ifftshift(Hc(t*Tkk(i)/kt)))/pi;
    subplot(121)
    semilogx(x,Tkk(i)*real(Hxcr*x0/pi)); hold on;
    subplot(122)
    plot(x,Tkk(i)*real(Hxcr*x0/pi)); hold on;
    if i==1
        k1 = sum(Hxcr)*dx/dt; 
    end
end
toc

Hc = Hc/k1;
%% COMPARE to a simple dft oversampling scheme:
xF = x(end);

pdfG = @(x) (x>=0).*(x.^(k-1)).*exp(-x/th)/(gamma(k)*(th^k));
% pdfU = @(x) ((x>=0).*(x<=x02))/(2*x0); # This seems to match analytic results less well.
pdfU = @(x) ((x>0).*(x<xF) + 0.5*((x==0) + (x==xF)))/(2*x0);
pdfH = @(x) pdfU(x).*pdfG(x)*(2*x0);

plot(x,pdfH(x))

nT = 3000;
tic
Ht_dft = fftshift(fft(pdfH(x),nT*(length(x)-1) + 1))*dx;
toc
plot(t,real(Ht_dft(1:nT:end))); hold on;
plot(t,imag(Ht_dft(1:nT:end)));

%%
Tkk = [1.0 1/3 0.10 1/30 0.01 1/300 0.001]; % seems to work well to ~1%
% Tkk = [1.0 0.33 0.10];
Tkkn = round(nT*Tkk);

nmid = nT*(Nt/2) + 1; % nb note not starting at 0!
figure;
for i = 1:numel(Tkk)
    Nstt = nmid - (Nt*Tkkn(i)/2);
    Nend = nmid + (Nt*Tkkn(i)/2);
    
    Hxcr = ifft(ifftshift( Ht_dft(Nstt:Tkkn(i):Nend) ))/pi;
    subplot(121)
    semilogx(x,Tkk(i)*real(Hxcr*x0/pi)); hold on;
    subplot(122)
    plot(x,Tkk(i)*real(Hxcr*x0/pi)); hold on;
%     sum(real(Hxcr))*dx/dt
end

%%
Tkkn_ni = Tkkn*3/pi;
figure;
for i = 1:numel(Tkk)
    Nstt = nmid - (Nt*Tkkn_ni(i)/2);
    Nend = nmid + (Nt*Tkkn_ni(i)/2);
    Tq = (Nstt:Tkkn_ni(i):Nend);
%     Ht_interp = interp1(Ht_dft,Tq);
%     Hxcr = ifft(ifftshift( Ht_interp ))/pi;
    Hxcr = ifft(ifftshift( Ht_dft(round(Tq)) ))/pi;
    subplot(121)
    semilogx(x,Tkkn_ni(i)*real(Hxcr*x0/pi)); hold on;
    subplot(122)
    plot(x,Tkkn_ni(i)*real(Hxcr*x0/pi)); hold on;
%     sum(real(Hxcr))*dx/dt
end

%%
tic
for i = 1:20000
    Ht_dft = fftshift(fft(pdfH(x)));
end 
toc
%%
dtFull = dt*0.5;
tFull = (-Dt/2:dtFull:Dt/2);
% dftFull = 






%% NOW: try adding together a bunch of these functions.
imax = 10;
Nadd = 10;
randnos = randi(imax,[1,Nadd])/imax;




%%A


A = [1,0,0;1,0.1,0;0,0,1] + 0.03*randn(3)
A
[U,S,V] = svd(A)


















