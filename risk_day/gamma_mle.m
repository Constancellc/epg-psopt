function [ a,b ] = gamma_mle( X )

X = X+1e-6; % get rid of any zeros
xb = mean(X);
A = logspace(-1,2,1e2);
B = A/xb;

% lL = zeros(size(A));
% for i = 1:numel(A)
%     lL_c = A(i)*log(B(i)) - log(gamma(A(i)));
%     for j = 1:numel(X)
%         lL(i) = lL(i) + lL_c + (A(i)-1)*log(X(j)) - B(i)*X(j);
%     end
% end
lL = sum( ones(size(X))*(A.*log(B) - log(gamma(A))) + ...
                        log(X)*(A - 1) - X.*B , 1 );
% semilogx(A,lL);

a = A(lL==max(lL));
b = B(lL==max(lL));
end

