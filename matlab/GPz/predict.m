function [mu,sigma,nu,beta_i,gamma,PHI,w,iSigma_w] = predict(X,model,varargin)
    
    method = model.method;
    
    pnames =    { 'whichSet' 'Psi' 'selection'};
    defaults =  { 'best'	[] true(size(X,1),1)};

    [whichSet,Psi,selection]  = internal.stats.parseArgs(pnames, defaults, varargin{:});
    
    if(strcmp(whichSet,'best'))
        set = model.best;
    else
        set = model.last;
    end
    
    n = sum(selection);
    k = model.k;
    m = model.m;
    
    muY = model.muY;
    muX = model.muX;
    sdX = model.sdX;

    
    X = X(selection,:);
    
    if(~isempty(Psi))
        if(method(2)=='C')
            Psi = Psi(:,:,selection);
        else
            Psi = Psi(selection,:);
        end
    end
    
    X = bsxfun(@minus,X,muX);
    X = bsxfun(@rdivide,X,sdX);
    
    
    w = set.w;
    iSigma_w = set.iSigma_w;
    
    
    Psi = fixPsi(Psi,n,sdX,method);
    
    missing = isnan(X);
    
    list = true(n,1);
    groups = logical([]);

    while(sum(list)>0)
        first = find(list,1);
        group = false(n,1);
        group(list) = sum(abs(bsxfun(@minus,missing(list,:),missing(first,:))),2)==0;
        groups = [groups group];
        list(group)=false;
    end

    mu = zeros(n,k);nu = zeros(n,k);beta_i = zeros(n,k);gamma = zeros(n,k);PHI = zeros(n,m);
    
    for i=1:size(groups,2)
        
        group = groups(:,i);
        
        if(method(2)=='C')
            [mu(group,:),nu(group,:),beta_i(group,:),gamma(group,:),PHI(group,:)] = predictCov(X,Psi,model,set,group);
        else
            [mu(group,:),nu(group,:),beta_i(group,:),gamma(group,:),PHI(group,:)] = predictDiag(X,Psi,model,set,group);
        end
    end
    
    
    sigma = nu+beta_i+gamma;
    mu = bsxfun(@plus,mu,muY);
    
    
end