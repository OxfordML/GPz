function [mu,sigma,nu,beta_i,gamma,PHI,w,iSigma_w] = predict(X,model,varargin)
    
    n = size(X,1);
    
    muY = model.muY;
    muX = model.muX;
    sdX = model.sdX;

    method = model.method;
    
    pnames =    { 'whichSet' 'Psi' 'exact'};
    defaults =  { 'best'	[] true};

    [whichSet,Psi,exact]  = internal.stats.parseArgs(pnames, defaults, varargin{:});
    
    if(strcmp(whichSet,'best'))
        set = model.best;
    else
        set = model.last;
    end
    
    X = bsxfun(@minus,X,muX);
    X = bsxfun(@rdivide,X,sdX);
    
    Psi = fixSx(Psi,n,sdX,method);
    
    theta = set.theta;
    w = set.w;
    iSigma_w = set.iSigma_w;
    
    [PHI,~,lnBeta_i] = getPHI(X,Psi,theta,model,[]);
    
    mu = PHI*w;
    beta_i = exp(lnBeta_i);
    
    [n,k] = size(mu);

    gamma = zeros(n,k);
    nu = zeros(n,k);
    
    if(~isempty(Psi)&&exact)

        switch(method(2))

            case 'C'
            
                [nu,Vg,gamma] = predictCov(X,Psi,model,set,PHI,mu,lnBeta_i);
            
            otherwise
                
                [nu,Vg,gamma] = predictDiag(X,Psi,model,set,PHI,mu,lnBeta_i);
        end
        
        beta_i = beta_i.*(1+Vg);
        
    else
        
        for o=1:k
            nu(:,o) = sum(PHI.*(PHI*iSigma_w(:,:,o)),2);
        end
    end

    sigma = gamma+nu+beta_i;

    mu = bsxfun(@plus,mu,muY);
    
end