normalDist = @(x, mu, s) exp(-.5*(x-mu).^2/s^2) / (s*sqrt(2*pi));

dom = [0, 1];
x = chebfun('x',dom,'splitting','on');
k = 0.66;

aB = 2; bB=5; B = beta(aB,bB);
s = 1e-2;
nmDistPst = @(x) 2*( exp(-.5*(x).^2/s^2) / (s*sqrt(2*pi)) ); % only defined for +ve x so needs to be doubled
pulse = @(x,x0,w) (heaviside(x-x0)-heaviside(x-x0-w))/w;
nmRect = @(x) (1-heaviside(x-s))/s

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

M = randi([-20,20],nConv,1)/4;
% M = ones(nConv,1);

tI = zeros(size(nConv));
lenN = zeros(size(nConv));
for i = 1:numel(nConv)
    nconv = nConv(i);
    
    kf_j = k;
    F_j = mixNoPulse(x,M(1));
    tic
    for j = 2:nconv
        
        domM = sort(dom*M(j));
        xM = chebfun('x',domM,'splitting','on');
        G_m = mixNoPulse(xM,1/M(j))/abs(M(j));
        kg_g = k;
        
        F_j,G_m = pad_domains(F_j,G_m);
        
        F_j = kf_j*G_m + kg_g*F_j + conv(F_j,G_m);
        kf_j = kf_j*kg_g;
        
        F_j = chebfun(F_j,'splitting','on','resampling','on');
        
        % Nj = conv(Nj,Nm);
        % Nj = conv(Nj,Nm,'old');
        % Nj = chebfun(Nj,'splitting','on','resampling','on');
    end
    tI(i) = toc;
    lenN(i) = length(Nj);
end

plot(F_j)

tI
lenN




function [Fn,Gn] = pad_domains(F,G)
dom1 = domain(F);
dom2 = domain(G);
a = dom1(1);
b = dom1(2);
c = dom2(1);
d = dom2(2);

if a<c
    zeroSet = chebfun('0',[a,c]);
    Gn = join(zeroSet,Gn);
end
if c>a
    zeroSet = chebfun('0',[c,a]);
    Fn = join(zeroSet,Fn);
end
if b<d
    zeroSet = chebfun('0',[b,d]);
    Fn = join(Fn,zeroSet);
end
if d<b
    zeroSet = chebfun('0',[d,b]);
    Gn = join(Gn,zeroSet);
end
end
