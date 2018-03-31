function [mu,nu,beta_i,gamma,PHI] = predictDiag(X,Psi,model,set,ind)
    
    o = ~isnan(X(find(ind,1),:));
    do = sum(o);
    
    d = size(X(ind,:),2);
    m = model.m;
    k = model.k;
    
    method = model.method;

    theta = set.theta;
    w = set.w;
    iSigma_w = set.iSigma_w;
    priors = set.priors;
    P = set.P;
    
    if(model.heteroscedastic)
        v = set.v;
    else
        v = zeros(m,k);
    end

    switch(method)
                        
        case 'GL'
            Gamma  = repmat(theta(m*d+1),m,d);
        case 'VL'
            Gamma  = repmat(theta(m*d+1:m*d+m),1,d);
        case 'GD'
            Gamma  = repmat(theta(m*d+1:m*d+d)',m,1);
        case 'VD'
            Gamma = reshape(theta(m*d+1:m*d+m*d),m,d);
    end

    g_dim = model.g_dim;
    b = theta(m*d+g_dim+m*k+1:m*d+g_dim+m*k+k)';
    
    if(do==d)
        
        if(isempty(Psi))
            
            [mu,nu,beta_i,gamma,PHI] = predictFull(X(ind,:),theta,w,iSigma_w,model);
            
        else
            [mu,nu,beta_i,gamma,PHI] = predictNoisy(X(ind,:),Psi(ind,:),Gamma,w,v,b,P,iSigma_w,theta,model);
        end
    else
        
        if(isempty(Psi))
            [mu,nu,beta_i,gamma,PHI] = predictMissing(X(ind,:),Gamma,w,v,b,P,iSigma_w,priors);
        else
            [mu,nu,beta_i,gamma,PHI] = predictNoisyMissing(X(ind,:),Psi(ind,:),Gamma,w,v,b,P,iSigma_w,priors);
        end
        
    end

function [mu,nu,beta_i,gamma,PHI] = predictFull(X,theta,w,iSigma_w,model)

    [n,~] = size(X);
    [~,k] = size(w);
    
    [PHI,~,ElnS] = getPHI(X,[],theta,model,[]);

    mu = PHI*w;

    nu = zeros(n,k);

    for out=1:k
        nu(:,out) = sum(PHI.*(PHI*iSigma_w(:,:,out)),2);
    end

    beta_i = exp(ElnS);
    gamma = zeros(n,k);
function [mu,nu,beta_i,gamma,PHI] = predictNoisy(X,Psi,Gamma,w,v,b,P,iSigma_w,theta,model)

    [n,d] = size(X);
    [m,k] = size(w);
    
    [PHI,~,ElnS] = getPHI(X,Psi,theta,model,[]);
    
    mu = PHI*w;
    
    nu = zeros(n,k);

    gamma = zeros(n,k);
    VlnS = zeros(n,k);
    
    iSigma = Gamma.^2;
    Sigma = Gamma.^-2;
    lnz = -0.5*sum(log(iSigma),2);
    
    for i=1:m
        for j=1:i


            iCij = iSigma(i,:)+iSigma(j,:);
            Cij = 1./iCij;
            cij = (P(i,:).*iSigma(i,:)+P(j,:).*iSigma(j,:))./iCij;

            lnZij = lnz(i)+lnz(j)-0.5*sum(power(P(i,:)-P(j,:),2)./(Sigma(i,:)+Sigma(j,:)))-0.5*sum(log(Sigma(i,:)+Sigma(j,:)));

            Delta = bsxfun(@minus,X,cij);

            Cij_plus_Psi = bsxfun(@plus,Psi,Cij);

            lnNxc = -0.5*sum((Delta.^2)./Cij_plus_Psi,2)-0.5*sum(log(Cij_plus_Psi),2);

            ZijNxc = exp(lnZij+lnNxc);
            
            gamma = gamma+2*ZijNxc*(w(i,:).*w(j,:));
            VlnS  = VlnS+2*ZijNxc*(v(i,:).*v(j,:));
            nu  = nu+2*ZijNxc*squeeze(iSigma_w(i,j,:))';

        end

        gamma = gamma-ZijNxc*(w(i,:).*w(j,:));
        VlnS  = VlnS-ZijNxc*(v(i,:).*v(j,:));
        nu  = nu-ZijNxc*squeeze(iSigma_w(i,j,:))';

    end

    VlnS = VlnS-bsxfun(@minus,ElnS,b).^2;
    gamma = gamma-mu.^2;
    beta_i = exp(ElnS).*(1+0.5*VlnS);
    
function [mu,nu,beta_i,gamma,PHI] = predictMissing(X,Gamma,w,v,b,P,iSigma_w,priors)
    
    o = ~isnan(X(1,:));
    u = ~o;
    
    [n,~] = size(X);
    [m,k] = size(w);
    
    iSigma = Gamma.^2;
    Sigma = Gamma.^-2;
    
    Ex = zeros(n,m);   
    
    lnz = -0.5*sum(log(iSigma),2)';
    
    No = zeros(n,m);
    
    for i=1:m
    
        Delta = bsxfun(@minus,X(:,o),P(i,o));
        No(:,i) = exp(-0.5*sum(bsxfun(@rdivide,Delta.^2,Sigma(i,o)),2)-0.5*sum(log(Sigma(i,o))));
        Ex(:,i) = No(:,i)*priors(i);

    end

    Ey = sum(Ex,2);
    
    Pio = bsxfun(@rdivide,Ex,Ey);
    
    i = mod(0:(m*m-1),m)'+1;
    j = floor((0:(m*m-1))/m)'+1;
    

    Nij = exp(-0.5*sum(bsxfun(@rdivide,(P(i,~o)-P(j,~o)).^2,Sigma(i,~o)+Sigma(j,~o)),2)-0.5*sum(log(Sigma(i,~o)+Sigma(j,~o)),2));
    
    PHI = bsxfun(@times,No(:,i).*Pio(:,j),Nij')*sparse(1:length(i),i,1,length(i),m);
    PHI = bsxfun(@times,PHI,exp(lnz));
    
    mu = PHI*w;
    ElnS = PHI*v;
    
    gamma = zeros(n,k);
    nu = zeros(n,k);
    VlnS = zeros(n,k);
    
    for i=1:m
        for j=1:i
            
            Cij = 1./(iSigma(i,:)+iSigma(j,:));
            cij = (P(i,:).*iSigma(i,:)+P(j,:).*iSigma(j,:)).*Cij;
            
            Delta = bsxfun(@minus,X(:,o),cij(o));
            No = exp(-0.5*sum(bsxfun(@rdivide,Delta.^2,Cij(o)),2)-0.5*sum(log(Cij(o))));
            
            Delta = bsxfun(@minus,P(:,u),cij(u));
            Cij_plus_Sigma = bsxfun(@plus,Sigma(:,u),Cij(u));
            Nu = exp(-0.5*sum(bsxfun(@rdivide,Delta.^2,Cij_plus_Sigma),2)-0.5*sum(log(Cij_plus_Sigma),2));
            
            N = No*Nu';
            EcCij = sum(N.*Pio,2);

            Delta = P(i,:)-P(j,:);
            Zij = exp(lnz(i)+lnz(j)-0.5*sum((Delta.^2)./(Sigma(i,:)+Sigma(j,:)))-0.5*sum(log(Sigma(i,:)+Sigma(j,:))))*EcCij;
            
            gamma = gamma+2*Zij*(w(i,:).*w(j,:));
            VlnS = VlnS+2*Zij*(v(i,:).*v(j,:));
            nu = nu+2*Zij*squeeze(iSigma_w(i,j,:))';
        end
        
        gamma = gamma-Zij*(w(i,:).*w(i,:));
        VlnS = VlnS-Zij*(v(i,:).*v(i,:));
        nu = nu-Zij*squeeze(iSigma_w(i,i,:))';

    end
    
    
    VlnS = VlnS-ElnS.^2;
   
    ElnS = bsxfun(@plus,ElnS,b);

    beta_i = exp(ElnS).*(1+0.5*VlnS);

    gamma = gamma-mu.^2;
    
function [mu,nu,beta_i,gamma,PHI] = predictNoisyMissing(X,Psi,Gamma,w,v,b,P,iSigma_w,priors)

    o = ~isnan(X(1,:));
    u = ~o;
    
    [n,~] = size(X);
    [m,k] = size(w);

    Ex = zeros(n,m);   
    
    iSigma = Gamma.^2;
    Sigma = Gamma.^-2;
    lnz = 0.5*sum(log(Sigma),2)';
    
    No = zeros(n,m);
    
    for i=1:m
    
        Delta = bsxfun(@minus,X(:,o),P(i,o));
        Sigma_plus_Psi = bsxfun(@plus,Psi,Sigma(i,:));
        No(:,i) = exp(-0.5*sum((Delta.^2)./Sigma_plus_Psi(:,o),2)-0.5*sum(log(Sigma_plus_Psi(:,o)),2));
        Ex(:,i) = No(:,i)*priors(i);

    end

    Ey = sum(Ex,2);
    
    Pio = bsxfun(@rdivide,Ex,Ey);
    
    i = mod(0:(m*m-1),m)'+1;
    j = floor((0:(m*m-1))/m)'+1;
    

    Nij = exp(-0.5*sum(bsxfun(@rdivide,(P(i,~o)-P(j,~o)).^2,Sigma(i,~o)+Sigma(j,~o)),2)-0.5*sum(log(Sigma(i,~o)+Sigma(j,~o)),2));
    
    PHI = bsxfun(@times,No(:,i).*Pio(:,j),Nij')*sparse(1:length(i),i,1,length(i),m);
    PHI = bsxfun(@times,PHI,exp(lnz));
    
    mu = PHI*w;
    ElnS = PHI*v;
    
    gamma = zeros(n,k);
    nu = zeros(n,k);
    VlnS = zeros(n,k);
    
    
    for i=1:m
        for j=1:i

            Cij = 1./(iSigma(i,:)+iSigma(j,:));
            cij = (P(i,:).*iSigma(i,:)+P(j,:).*iSigma(j,:)).*Cij;
            
            Delta = bsxfun(@minus,X(:,o),cij(o));
            Cij_plus_Psi = bsxfun(@plus,Psi(:,o),Cij(o));
            No = exp(-0.5*sum((Delta.^2)./Cij_plus_Psi,2)-0.5*sum(log(Cij_plus_Psi),2));
            
            Delta = bsxfun(@minus,P(:,u),cij(u));
            Cij_plus_Psi = bsxfun(@plus,Sigma(:,u),Cij(u));
            Nu = exp(-0.5*sum((Delta.^2)./Cij_plus_Psi,2)-0.5*sum(log(Cij_plus_Psi),2));
            
            N = No*Nu';
            EcCij = sum(N.*Pio,2);
            
            Delta = P(i,:)-P(j,:);
            Pij = exp(lnz(i)+lnz(j)-0.5*sum((Delta.^2)./(Sigma(i,:)+Sigma(j,:)))-0.5*sum(log(Sigma(i,:)+Sigma(j,:))))*EcCij;
            
            gamma = gamma+2*Pij*(w(i,:).*w(j,:));
            VlnS = VlnS+2*Pij*(v(i,:).*v(j,:));
            nu = nu+2*Pij*squeeze(iSigma_w(i,j,:))';
        end
        
        gamma = gamma-Pij*(w(i,:).*w(j,:));
        VlnS = VlnS-Pij*(v(i,:).*v(j,:));
        nu = nu-Pij*squeeze(iSigma_w(i,j,:))';

    end
    
    
    VlnS = VlnS-ElnS.^2;
   
    ElnS = bsxfun(@plus,ElnS,b);

    beta_i = exp(ElnS).*(1+0.5*VlnS);

    gamma = gamma-mu.^2;
    

