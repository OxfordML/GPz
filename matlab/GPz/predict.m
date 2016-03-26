function [mu,sigma,modelV,noiseV,PHI] = predict(X,model,whichSet)
    
    if(nargin<3||strcmp(whichSet,'best'))
        set = model.best;
    else
        set = model.last;
    end
    
    theta = set.theta;
    w = set.w;
    SIGMAi = set.SIGMAi;
    
    muY = model.muY;
    
    X = bsxfun(@minus,X,model.muX)*model.T;
    
    if(strcmp(model.method,'ANN'))
        [~,~,~,~,PHI,lnBeta] = ANN(theta,model,X,[],[],[],[]);
    else
        [~,~,~,~,PHI,lnBeta] = GPz(theta,model,X,[],[],[],[]);
    end
    
    mu = bsxfun(@plus,PHI*w,muY);

    [n,k] = size(mu);
    
    modelV = zeros(n,k);
    for i=1:k
        modelV(:,i) = sum(PHI.*(PHI*SIGMAi(:,:,i)),2);
    end
    
    noiseV = exp(-lnBeta);
    sigma = modelV+noiseV;
    
end