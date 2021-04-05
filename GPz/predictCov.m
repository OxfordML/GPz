function [mu,nu,beta_i,gamma,PHI] = predictCov(X,Psi,model,set,ind)
    
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
        case 'GC'
            Gamma  = reshape(repmat(reshape(theta(m*d+1:m*d+d*d),d,d),1,m),d,d,m);
        case 'VC'
            Gamma = reshape(theta(m*d+1:m*d+d*d*m),d,d,m);
    end

    g_dim = model.g_dim;
    b = theta(m*d+g_dim+m*k+1:m*d+g_dim+m*k+k)';
    
    if(do==d)
        
        if(isempty(Psi))
            
            [mu,nu,beta_i,gamma,PHI] = predictFull(X(ind,:),theta,w,iSigma_w,model);
            
        else
            [mu,nu,beta_i,gamma,PHI] = predictNoisy(X(ind,:),Psi(:,:,ind),Gamma,w,v,b,P,iSigma_w,theta,model);
        end
    else
        
        if(isempty(Psi))
            [mu,nu,beta_i,gamma,PHI] = predictMissing(X(ind,:),Gamma,w,v,b,P,iSigma_w,priors);
        else
            [mu,nu,beta_i,gamma,PHI] = predictNoisyMissing(X(ind,:),Psi(:,:,ind),Gamma,w,v,b,P,iSigma_w,priors);
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
    
    iSigma = zeros(d,d,m);
    Sigma = zeros(d,d,m);
    lnz = zeros(m,1);
    
    [PHI,~,ElnS] = getPHI(X,Psi,theta,model,[]);
   
    mu = PHI*w;
    
    nu = zeros(n,k);
    VlnS = zeros(n,k);
    gamma = zeros(n,k);
    
    for i=1:m

        iSigma(:,:,i) = Gamma(:,:,i)'*Gamma(:,:,i);
        Sigma(:,:,i) = inv(iSigma(:,:,i));

        lnz(i) = -0.5*sum(log(svd(iSigma(:,:,i))));
    end
    
    for id=1:n
        
        for i=1:m

            for j=1:i

                iCij = iSigma(:,:,i)+iSigma(:,:,j);
                Cij = inv(iCij);
                cij = (P(i,:)*iSigma(:,:,i)+P(j,:)*iSigma(:,:,j))/iCij;

                Delta = P(i,:)-P(j,:);

                lnZij = lnz(i)+lnz(j)-0.5*(Delta/(Sigma(:,:,i)+Sigma(:,:,j)))*Delta'-0.5*sum(log(svd(Sigma(:,:,i)+Sigma(:,:,j))));

                Delta = X(id,:)-cij;

                Cij_plus_Psi = Psi(:,:,id)+Cij;

                lnNxc = -0.5*(Delta/Cij_plus_Psi)*Delta'-0.5*sum(log(svd(Cij_plus_Psi)));
                
                ZijNxc = exp(lnZij+lnNxc);

                gamma(id,:) = gamma(id,:)+2*ZijNxc*(w(i,:).*w(j,:));
                VlnS(id,:)  = VlnS(id,:)+2*ZijNxc*(v(i,:).*v(j,:));
                nu(id,:)  = nu(id,:)+2*ZijNxc*squeeze(iSigma_w(i,j,:))';

            end

            gamma(id,:) = gamma(id,:)-ZijNxc*(w(i,:).*w(j,:));
            VlnS(id,:)  = VlnS(id,:)-ZijNxc*(v(i,:).*v(j,:));
            nu(id,:)  = nu(id,:)-ZijNxc*squeeze(iSigma_w(i,j,:))';

        end

    end
    
    VlnS = VlnS-bsxfun(@minus,ElnS,b).^2;
    gamma = gamma-mu.^2;
    beta_i = exp(ElnS).*(1+0.5*VlnS);
function [mu,nu,beta_i,gamma,PHI] = predictMissing(X,Gamma,w,v,b,P,iSigma_w,priors)
    
    o = ~isnan(X(1,:));
    do = sum(o);
    du = sum(~o);
    
    [n,d] = size(X);
    [m,k] = size(w);
    
    PHI = zeros(n,m);
    
    gamma = zeros(n,k);
    nu = zeros(n,k);
    VlnS = zeros(n,k);
    
    Ex = zeros(n,m);   
    
    R = zeros(do,du,m);
    Psi_hat = zeros(d,d,m);
    X_hat = zeros(n,d,m);
    lnz = zeros(1,m);

    Sigma = zeros(d,d,m);
    iSigma = zeros(d,d,m);
    
    for i=1:m
 
        iSigma(:,:,i) = Gamma(:,:,i)'*Gamma(:,:,i);
        Sigma(:,:,i)  = inv(iSigma(:,:,i));
        
        lnz(i) = -0.5*sum(log(svd(iSigma(:,:,i))));

        Delta = bsxfun(@minus,X(:,o),P(i,o));
        Ex(:,i) = exp(-0.5*sum((Delta/Sigma(o,o,i)).*Delta,2)-0.5*sum(log(svd(Sigma(o,o,i)))))*priors(i);
        
        R(:,:,i) = Sigma(o,o,i)\Sigma(o,~o,i);
        
        Psi_hat(~o,~o,i) = Sigma(~o,~o,i)-Sigma(~o,o,i)*R(:,:,i);
        
        X_hat(:,~o,i) = bsxfun(@plus,bsxfun(@minus,X(:,o),P(i,o))*R(:,:,i),P(i,~o));
        X_hat(:,o,i) = X(:,o);
    end

    
    Pio = bsxfun(@rdivide,Ex,sum(Ex,2));
    
    for i=1:m
        for j=1:i

            iCij = iSigma(:,:,i)+iSigma(:,:,j);
            Cij = inv(iCij);
            cij = (P(i,:)*iSigma(:,:,i)+P(j,:)*iSigma(:,:,j))/iCij;
            
            Delta = bsxfun(@minus,X_hat(:,:,j),P(i,:));
            N = exp(-0.5*sum((Delta/(Sigma(:,:,i)+Psi_hat(:,:,j))).*Delta,2)-0.5*sum(log(svd(Sigma(:,:,i)+Psi_hat(:,:,j)))));
            NPio = N.*Pio(:,j);
            PHI(:,i) = PHI(:,i)+NPio;
            
            Delta = bsxfun(@minus,X_hat(:,:,i),P(j,:));
            N = exp(-0.5*sum((Delta/(Sigma(:,:,j)+Psi_hat(:,:,i))).*Delta,2)-0.5*sum(log(svd(Sigma(:,:,j)+Psi_hat(:,:,i)))));
            NPio = N.*Pio(:,i);
            PHI(:,j) = PHI(:,j)+NPio;

            EcCij = zeros(n,1);
            for l=1:m
                Delta = bsxfun(@minus,X_hat(:,:,l),cij);
                N = exp(-0.5*sum((Delta/(Cij+Psi_hat(:,:,l))).*Delta,2)-0.5*sum(log(svd(Cij+Psi_hat(:,:,l)))));
                EcCij = bsxfun(@plus,N.*Pio(:,l),EcCij);
            end

            Delta = P(i,:)-P(j,:);
            Zij = exp(lnz(i)+lnz(j)-0.5*(Delta/(Sigma(:,:,i)+Sigma(:,:,j)))*Delta'-0.5*sum(log(svd(Sigma(:,:,i)+Sigma(:,:,j)))))*EcCij;
            
            gamma = gamma+2*Zij*(w(i,:).*w(j,:));
            VlnS = VlnS+2*Zij*(v(i,:).*v(j,:));
            nu = nu+2*Zij*squeeze(iSigma_w(i,j,:))';
        end
        
        PHI(:,i) = PHI(:,i)-NPio;
        
        gamma = gamma-Zij*(w(i,:).*w(j,:));
        VlnS = VlnS-Zij*(v(i,:).*v(j,:));
        nu = nu-Zij*squeeze(iSigma_w(i,j,:))';

    end
    
    PHI = bsxfun(@times,PHI,exp(lnz));
    
    mu = PHI*w;
    ElnS = PHI*v;
    
    
    VlnS = VlnS-ElnS.^2;
   
    ElnS = bsxfun(@plus,ElnS,b);

    beta_i = exp(ElnS).*(1+0.5*VlnS);

    gamma = gamma-mu.^2;
function [mu,nu,beta_i,gamma,PHI] = predictNoisyMissing(X,Psi,Gamma,w,v,b,P,iSigma_w,priors)

    o = ~isnan(X(1,:));
    do = sum(o);
    du = sum(~o);
    
    [n,d] = size(X);
    [m,k] = size(w);
    
    PHI = zeros(n,m);
    gamma = zeros(n,k);
    nu = zeros(n,k);
    VlnS = zeros(n,k);
    
    Ex = zeros(n,m);   
    
    R = zeros(do,du,m);
    Psi_hat = zeros(d,d,m,n);
    X_hat = zeros(n,d,m);
    lnz = zeros(1,m);

    Sigma = zeros(d,d,m);
    iSigma = zeros(d,d,m);
    
    for id=1:n
        for i=1:m

            iSigma(:,:,i) = Gamma(:,:,i)'*Gamma(:,:,i);

            Sigma(:,:,i)  = inv(iSigma(:,:,i));
            lnz(i) = -0.5*sum(log(svd(iSigma(:,:,i))));

            Delta = bsxfun(@minus,X(id,o),P(i,o));
            Ex(id,i) = exp(-0.5*sum((Delta/(Sigma(o,o,i)+Psi(o,o,id))).*Delta,2)-0.5*sum(log(svd(Sigma(o,o,i)+Psi(o,o,id)))))*priors(i);

            R(:,:,i) = Sigma(o,o,i)\Sigma(o,~o,i);
            T = [eye(do);R(:,:,i)'];

            [~,unshuffle] = sort([find(o) find(~o)]);

            Psi_hat(unshuffle,unshuffle,i,id) = T*Psi(o,o,id)*T';
            Psi_hat(~o,~o,i,id) = Psi_hat(~o,~o,i,id)+Sigma(~o,~o,i)-Sigma(~o,o,i)*R(:,:,i);

            X_hat(id,~o,i) = bsxfun(@plus,bsxfun(@minus,X(id,o),P(i,o))*R(:,:,i),P(i,~o));
            X_hat(id,o,i) = X(id,o);
        end
    end

    Pio = bsxfun(@rdivide,Ex,sum(Ex,2));
    
    for id=1:n
        for i=1:m
            for j=1:i

                iCij = iSigma(:,:,i)+iSigma(:,:,j);
                Cij = inv(iCij);
                cij = (P(i,:)*iSigma(:,:,i)+P(j,:)*iSigma(:,:,j))/iCij;

                Delta = bsxfun(@minus,X_hat(id,:,j),P(i,:));
                N = exp(-0.5*sum((Delta/(Sigma(:,:,i)+Psi_hat(:,:,j,id))).*Delta,2)-0.5*sum(log(svd(Sigma(:,:,i)+Psi_hat(:,:,j,id)))));
                NPio = N*Pio(id,j);
                PHI(id,i) = bsxfun(@plus,NPio,PHI(id,i));

                Delta = bsxfun(@minus,X_hat(id,:,i),P(j,:));
                N = exp(-0.5*sum((Delta/(Sigma(:,:,j)+Psi_hat(:,:,i,id))).*Delta,2)-0.5*sum(log(svd(Sigma(:,:,j)+Psi_hat(:,:,i,id)))));
                NPio = N*Pio(id,i);
                PHI(id,j) = bsxfun(@plus,NPio,PHI(id,j));

                EcCij = 0;
                for l=1:m
                    Delta = bsxfun(@minus,X_hat(id,:,l),cij);
                    N = exp(-0.5*sum((Delta/(Cij+Psi_hat(:,:,l,id))).*Delta,2)-0.5*sum(log(svd(Cij+Psi_hat(:,:,l,id)))));
                    EcCij = bsxfun(@plus,N.*Pio(id,l),EcCij);
                end

                Delta = P(i,:)-P(j,:);
                Pij = exp(lnz(i)+lnz(j)-0.5*(Delta/(Sigma(:,:,i)+Sigma(:,:,j)))*Delta'-0.5*sum(log(svd(Sigma(:,:,i)+Sigma(:,:,j)))))*EcCij;

                gamma(id,:) = gamma(id,:)+2*Pij*(w(i,:).*w(j,:));
                VlnS(id,:) = VlnS(id,:)+2*Pij*(v(i,:).*v(j,:));
                nu(id,:) = nu(id,:)+2*Pij*squeeze(iSigma_w(i,j,:))';
            end

            PHI(id,j) = PHI(id,i)-NPio;

            gamma(id,:) = gamma(id,:)-Pij*(w(i,:).*w(j,:));
            VlnS(id,:) = VlnS(id,:)-Pij*(v(i,:).*v(j,:));
            nu(id,:) = nu(id,:)-Pij*squeeze(iSigma_w(i,j,:))';

        end
    end
    
    PHI = bsxfun(@times,PHI,exp(lnz));
    
    mu = PHI*w;
    ElnS = PHI*v;
    
    VlnS = VlnS-ElnS.^2;
   
    ElnS = bsxfun(@plus,ElnS,b);

    beta_i = exp(ElnS).*(1+0.5*VlnS);

    gamma = gamma-mu.^2;
    

