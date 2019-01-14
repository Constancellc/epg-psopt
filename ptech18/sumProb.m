normalDist = @(x, mu, s) exp(-.5*(x-mu).^2/s^2) / (s*sqrt(2*pi));

dom = [0, 1];
x = chebfun('x',dom,'splitting','on');
k = 0.5;

aB = 2; bB=5; B = beta(aB,bB);
s = 1e-2;
nmDistPst = @(x) 2*( exp(-.5*(x).^2/s^2) / (s*sqrt(2*pi)) ); % only defined for +ve x so needs to be doubled
pulse = @(x,x0,w) (heaviside(x-x0)-heaviside(x-x0-w))/w;
nmRect = @(x) (1-heaviside(x-s))/s;

w = 1e-2;
twoPulse = k*pulse(x,0,w) + (1-k)*pulse(x,1-w,w);


% mixDist = @(x,m) k*nmRect(x*m) + (1-k)*( ((x*m).^(aB-1)).*((1-(x*m)).^(bB-1))/B );
mixDist = @(x,m) k*nmDistPst(x*m) + (1-k)*( ((x*m).^(aB-1)).*((1-(x*m)).^(bB-1))/B );

mixNoPulse = @(x,m) (1-k)*( ((x*m).^(aB-1)).*((1-(x*m)).^(bB-1))/B );

N0 = mixDist(x,1);

% kM = -0.5;
% domM = sort(dom*kM);
% xM = chebfun('x',domM,'splitting','on');
% Nm = mixDist(xM,1/kM)/abs(kM);
% plot(N0); hold on; plot(Nm)


nConv = round(logspace(0,1.8,8));
nConv = 20;

M = (randi([-20,20],nConv,1)/4) + 0.125;
% M = ones(nConv,1);

tI = zeros(size(nConv));
lenN = zeros(size(nConv));
for i = 1:numel(nConv)
    nconv = nConv(i);
    
    kf_j = k;
    dom1 = sort(dom*M(1));
    x1 = chebfun('x',dom1,'splitting','on');
    F_j = mixNoPulse(x1,1/M(1))/abs(M(1));
    tic
    for j = 2:nconv
        
        domM = sort(dom*M(j));
        xM = chebfun('x',domM,'splitting','on');
        G_m = mixNoPulse(xM,1/M(j))/abs(M(j));
        kg_g = k;
        
        
        FGconv = conv(F_j,G_m,'old');
        [FGconv,G_m] = pad_domains(FGconv,G_m);
        [FGconv,F_j] = pad_domains(FGconv,F_j);
        [FGconv,G_m] = pad_domains(FGconv,G_m);
        
        F_j = kf_j*G_m + kg_g*F_j + FGconv;
        kf_j = kf_j*kg_g;
        
        F_j = chebfun(F_j,'splitting','on','resampling','on');
        
        % Nj = conv(Nj,Nm);
        % Nj = conv(Nj,Nm,'old');
        % Nj = chebfun(Nj,'splitting','on','resampling','on');
    end
    tI(i) = toc;
    lenN(i) = length(F_j);
end

plot(F_j)

tI
lenN

dom1 = domain(F_j);
dom1 = domain(FGconv);
dom2 = domain(G_m);
a = dom1(1);
b = dom1(end);
c = dom2(1);
d = dom2(end);

%%
% try again but only using numerical convolution?
k = 0.5;
aB = 2; bB=5; B = beta(aB,bB);
nConv = round(logspace(0,1.8,8));
nConv = round(logspace(0,2.5,12));
% nConv = [20;



mixDist = @(x,m,dx) (k*(x==0)/(dx*abs(m))) + (1-k)*( ((x*m).^(aB-1)).*((1-(x*m)).^(bB-1))/B );

x = linspace(0,1,1e4);
dx=x(2)-x(1);

f0 = mixDist(x,1,dx);
xM = linspace(0,M(1),abs(M(1))/dx);
fm = mixDist(xM,1/M(1),dx)/abs(M(1));

sum(f0)*dx;
sum(fm)*dx;

for i = 1:numel(nConv)
    nconv = nConv(i);
    M = (randi([-20,20],nConv(i),1)/4) + 0.125;
    
    x1 = linspace(0,M(1),1e3*abs(M(1)));
    Fj = mixDist(x1,1/M(1),dx)/abs(M(1));
    tic
    for j = 2:nconv
        
        xM = linspace(0,M(j),1e3*abs(M(j)));
        Gj = mixDist(xM,1/M(j),dx)/abs(M(j));
       
        Fj = 1e-3*conv(Fj,Gj);
    end
    tI(i) = toc;
    lenN(i) = length(Fj);
end
lenN
tI

%%

function [Fn,Gn] = pad_domains(F,G)
dom1 = domain(F);
dom2 = domain(G);
a = dom1(1);
b = dom1(end);
c = dom2(1);
d = dom2(end);

if a<c
    zeroSet = chebfun('0',[a,c]);
    Gn = join(zeroSet,G);
end
if c<a
    zeroSet = chebfun('0',[c,a]);
    Fn = join(zeroSet,F);
end
if b<d
    zeroSet = chebfun('0',[b,d]);
    Fn = join(F,zeroSet);
end
if d<b
    zeroSet = chebfun('0',[d,b]);
    Gn = join(G,zeroSet);
end

if exist('Fn','var')==0
    Fn = F;
end
if exist('Gn','var')==0
    Gn = G;
end

end



