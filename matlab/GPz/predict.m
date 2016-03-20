function [pred,sigma,modelV,noiseV,PHI] = predict(X,model,whichSet)
    
    if(nargin<3||strcmp(whichSet,'best'))
        set = model.best;
    else
        set = model.last;
    end
    
    method = model.method;
    
    theta = set.theta;
    w = set.w;
    SIGMAi = set.SIGMAi;
    
    joint = model.joint;
    heteroscedastic = model.heteroscedastic;
    muY = model.muY;
    
    m = model.m;
    
    X = bsxfun(@minus,X,model.muX)*model.T;
    
    [~,~,~,~,PHI,lnBeta] = GPz(theta,method,m,X,[],[],joint,heteroscedastic,[],[]);
    
    pred = PHI*w+muY;

    noiseV = exp(-lnBeta);
    modelV = sum(PHI.*(PHI*SIGMAi),2);
    sigma = modelV+noiseV;
    
end