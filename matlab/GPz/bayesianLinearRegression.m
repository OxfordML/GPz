function [cost,grad] = bayesianLinearRegression(lnAlpha,X,y)

    [n,d] = size(X);
    
    alpha = exp(lnAlpha);
    
    A = diag(alpha);
    
    SIGMA = X'*X+A;
    [U,S,V] = svd(SIGMA);
    
    SIGMAi = V*diag(1./diag(S))*U';
    w = SIGMAi*(X'*y);
    
    delta = X*w-y;
    
    cost = -0.5*sum(delta(:).^2)-0.5*sum(alpha.*w.^2)+0.5*sum(lnAlpha);
    
    dwda = -SIGMAi*(alpha.*w);
    
    grad = -(X'*delta).*dwda-alpha.*w.*dwda-0.5*alpha.*w.^2+0.5;   
    
    cost = -cost;
    grad = -grad;
    
end