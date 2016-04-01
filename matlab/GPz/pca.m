function [mu,T,Ti,U,S] = pca(X,th)


[n,m] = size(X);

missing = isnan(X);
X(missing) = 0;

counts = n-sum(missing);

mu = sum(X)./counts;

X = bsxfun(@minus,X,mu);
X(missing) = 0;

missing = double(missing);

sigmas = n*(X'*X)./(n-missing'*missing);

[U,S] = eig(sigmas);

S = abs(diag(S));

[~,order] = sort(-S);

U = U(:,order);
S = S(order);

c = cumsum(S/sum(S));
c(end)=1;

if(th==1)
    k=m;
else
    k = find(c>=th,1,'first');
end

S = sqrt(S(1:k)/(n-1));

S_inv = diag(1./S);
S = diag(S);

U = U(:,1:k);

T = U*S_inv;
Ti = S*U';

end
