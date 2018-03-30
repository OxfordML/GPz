function [PHI,Gamma,lnBeta_i,N] = getPHI(X,Psi,theta,model,selection)

    if(isempty(selection))
        selection = true(size(X,1),1);
    end

    n = sum(selection);
    d = model.d;
    m = model.m;
    k = model.k;

    method = model.method;
    
    X = X(selection,:);
    
    if(~isempty(Psi))
        if(method(2)=='C')
            Psi = Psi(:,:,selection);
        else
            Psi = Psi(selection,:);
        end
    end
    
    P = reshape(theta(1:m*d),m,d);
    
    switch(method)
                        
        case 'GL'
            Gamma  = repmat(theta(m*d+1),m,d);
        case 'VL'
            Gamma  = repmat(theta(m*d+1:m*d+m),1,d);
        case 'GD'
            Gamma  = repmat(theta(m*d+1:m*d+d)',m,1);
        case 'VD'
            Gamma = reshape(theta(m*d+1:m*d+m*d),m,d);
        case 'GC'
            Gamma  = reshape(repmat(reshape(theta(m*d+1:m*d+d*d),d,d),1,m),d,d,m);
        case 'VC'
            Gamma = reshape(theta(m*d+1:m*d+d*d*m),d,d,m);
    end
    
    
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
    
    lnPHI = zeros(n,m);
    lnN = zeros(n,m);
    
                
    for i=1:size(groups,2)

        group = groups(:,i);

        u = isnan(X(find(group,1),:));
        o = ~u;
            
        for j=1:m
            
            Delta = bsxfun(@minus,X(:,o),P(j, o));
            
            if(method(2)=='C')
                
                Sigma = inv(Gamma(:,:,j)'*Gamma(:,:,j));
                
                if(isempty(Psi))
                    lnPHI(group,j) = -0.5*sum((Delta(group,:)/Sigma(o,o)).*Delta(group,:),2)-0.5*sum(u)*log(2);
                    lnN(group,j) = lnPHI(group,j)-0.5*sum(log(svd(Sigma(o,o))))-0.5*sum(o)*log(2*pi)+0.5*sum(u)*log(2);
                else

                    index = find(group);
                    
                    for id=1:sum(group)
                        
                        PsiPlusSigma = Psi(o,o,index(id))+Sigma(o,o);

                        lnPHI(index(id),j) = -0.5*sum((Delta(index(id),:)/PsiPlusSigma).*Delta(index(id),:),2)+0.5*sum(log(svd(Sigma(o,o))))-0.5*sum(log(svd(PsiPlusSigma)))-0.5*sum(u)*log(2);
                        lnN(index(id),j) = lnPHI(index(id),j)-0.5*sum(log(svd(Sigma(o,o))))-0.5*sum(o)*log(2*pi)+0.5*sum(u)*log(2);
                    end
                end
                
            else
                
                Sigma = Gamma(j,o).^-2;
                
                if(isempty(Psi))

                    lnPHI(group,j) = -0.5*sum((bsxfun(@rdivide,Delta(group,:).^2,Sigma)),2)-0.5*sum(u)*log(2);
                    lnN(group,j) = lnPHI(group,j)-0.5*sum(log(Sigma))-0.5*sum(o)*log(2*pi)+0.5*sum(u)*log(2);

                else
                    
                    PsiPlusSigma = bsxfun(@plus,Psi(group,o),Sigma);
                    
                    lnPHI(group,j) = -0.5*sum((Delta(group,:).^2)./PsiPlusSigma,2)-0.5*sum(log(1+bsxfun(@rdivide,Psi(group,o),Sigma)),2)-0.5*sum(u)*log(2);
                    lnN(group,j) = lnPHI(group,j)-0.5*sum(log(Sigma))-0.5*sum(o)*log(2*pi)+0.5*sum(u)*log(2);

                end
            end
        end
    end
    
    
    PHI = exp(lnPHI);
    N = exp(lnN);
    
    g_dim = model.g_dim;
    b = theta(m*d+g_dim+m*k+1:m*d+g_dim+m*k+k)';
    
    lnBeta_i = repmat(b,n,1);

    if(model.heteroscedastic)
        v = reshape(theta(m*d+g_dim+m*k+k+1:m*d+g_dim+m*k+k+m*k),m,k);

        lnBeta_i = lnBeta_i+PHI*v;
    end
    
    
end