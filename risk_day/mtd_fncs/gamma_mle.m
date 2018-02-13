function [ a,b ] = gamma_mle( X )

X = X+1e-6; % get rid of any zeros
xb = mean(X);
A = logspace(-1,1,1e4);
B = A/xb;

lL = sum( ones(size(X))*(A.*log(B) - log(gamma(A))) + ...
                        log(X)*(A - 1) - X.*B , 1 );

a = A(lL==max(lL));
b = B(lL==max(lL));
end