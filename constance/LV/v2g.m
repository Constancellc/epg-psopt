% V2G optimization

% I want to see if the matrix is positive definite

n_ = 1;
t_ = 2;

M = zeros(2*n_*t_,2*n_*t_);
size(M)
for v=0:n_-1
    for t=0:t_-1
        M(v*t_+t+1,(v+n_)*t_+t+1) = 1;
        M((v+n_)*t_+t+1,v*t_+t+1) = 1;
    end
end

size(M)
eig(M)