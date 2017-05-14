function [mu,sigma,nu,beta_i,gamma,PHI,w,iSigma_w] = predict(X,model,varargin)
    
    [n,d] = size(X);
    
    muY = model.muY;
    muX = model.muX;
    sdX = model.sdX;

    method = model.method;
    
    learnPsi = model.learnPsi;
    
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
    
    theta = set.theta;
    w = set.w;
    iSigma_w = set.iSigma_w;
    
    if(learnPsi)
        if(method(2)=='C')
            S = reshape(theta(end-d*d+1:end),d,d);
            Psi = reshape(repmat(S'*S,1,n),d,d,n);
        else
            S = reshape(theta(end-d+1:end),1,d);
            Psi = ones(n,1)*S.^2;
        end            
    else
        Psi = fixSx(Psi,n,sdX,method);
    end
    
    
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