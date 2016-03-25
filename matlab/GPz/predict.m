function [pred,sigma,modelV,noiseV,PHI] = predict(X,model,whichSet)
    
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
    
    pred = PHI*w+muY;

    noiseV = exp(-lnBeta);
    modelV = sum(PHI.*(PHI*SIGMAi),2);
    sigma = modelV+noiseV;
    
end