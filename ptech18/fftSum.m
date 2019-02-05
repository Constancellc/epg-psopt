% fftSum is a script to implement the experimental work from fft_calcs for
% using chebfun convolution to calculate total PDFs.

Nt = 8e3; % This is about as high as we go.
% Nt = 2e3;
% Nt = 2e2;

th_k = 0.3;
th_k = 1.0; 
% th_k = 3.0;
% th_k = 30.0;

Dt = 2;
dt = Dt/Nt;

t = (-Dt/2:dt:Dt/2);
dx = 2*pi/Dt;
Dx = 2*pi/dt;
x = (-Dx/2:dx:Dx/2);
x = x + (Dx/2);

x0 = x(round(x0f*length(x)));
x02 = x(end);

% cf for gamma function
k = 2;
th = th_k/dt;

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
%% NOW: try adding together a bunch of these functions.
imax = 10;
Nadd = 10;
randnos = randi(imax,[1,Nadd])/imax;





