function [mu,sigma,nu,beta_i,PHI] = predict(X,model,whichSet)
    
    if(nargin<3||strcmp(whichSet,'best'))
        set = model.best;
    else
        set = model.last;
    end
    
    theta = set.theta;
    w = set.w;
    SIGMAi = set.SIGMAi;
    wL = model.wL;
    
    muY = model.muY;
    
    X = bsxfun(@minus,X,model.muX)*model.T;
    
    if(strcmp(model.method,'ANN'))
        [~,~,~,~,PHI,lnBeta] = ANN(theta,model,X,[],[],[],[]);
    else
        [PHI,lnBeta] = getPHI(X,1,theta,model);
    end
    
    mu = PHI*w;
    mu = bsxfun(@plus,mu,X*wL);
    mu = bsxfun(@plus,mu,muY);

    [n,k] = size(mu);
    
    nu = zeros(n,k);
    for i=1:k
        nu(:,i) = sum(PHI.*(PHI*SIGMAi(:,:,i)),2);
    end
    
    beta_i = exp(-lnBeta);
    sigma = nu+beta_i;
    
end