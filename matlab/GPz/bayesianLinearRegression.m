function [nlogML,grad,w] = bayesianLinearRegression(theta,X,Y)


[n,d] = size(X);

k = size(Y,2);

lnAlpha = reshape(theta(1:d*k),d,k);
lnBeta = reshape(theta(d*k+1:end),1,k);


beta = exp(lnBeta);
alpha = exp(lnAlpha);

nu = zeros(n,k);
w = zeros(d,k);
dwda = zeros(d,k);
SIGMAi = zeros(d,d,k);
logdet = zeros(1,k);
dlnAlpha = zeros(d,k);

for i=1:k
    
    BxX = beta(i)*X;

    SIGMA = BxX'*X+diag(alpha(:,i));

    [U,S] = svd(SIGMA);
    
    SIGMAi(:,:,i) = (U/S)*U';
    logdet(i) = sum(log(abs(diag(S))));
    
    nu(:,i) = sum(X.*(X*SIGMAi(:,:,i)),2);
              
    w(:,i) = SIGMAi(:,:,i)*BxX'*Y(:,i);

    dwda(:,i) = -SIGMAi(:,:,i)*(alpha(:,i).*w(:,i));
    
    dlnAlpha(:,i) = -0.5*diag(SIGMAi(:,:,i)).*alpha(:,i);

end

delta = X*w-Y;

beta_x_delta = bsxfun(@times,delta,beta);

nlogML = -0.5*sum(beta_x_delta.*delta)+0.5*n*lnBeta-0.5*sum(alpha.*w.^2)+0.5*sum(lnAlpha)-0.5*logdet-0.5*n*log(2*pi);

dlnAlpha = dlnAlpha-(X'*beta_x_delta).*dwda-alpha.*w.*dwda-0.5*alpha.*w.^2+0.5;

dlnBeta = -0.5*beta.*sum(delta.^2+nu)+0.5*n;

grad = [dlnAlpha(:);dlnBeta(:)];

nlogML = -sum(nlogML)/(n*k);
grad = -grad/(n*k);

end
