function [Xi,logdet] = inv_logdet(X)

[U,S,V] = svd(X,'econ');

s = diag(S);

tol = max(size(X)) * eps(norm(s,inf));

r1 = sum(s > tol)+1;
V(:,r1:end) = [];
U(:,r1:end) = [];
s(r1:end) = [];

Xi = bsxfun(@rdivide,V,s.')*U';
logdet = sum(log(s));
