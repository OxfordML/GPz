function [PHI,lnBeta,GAMMA] = getPHI(X,omega,theta,model)

    [n,d] = size(X);
    
    m = model.m;
    method = model.method;
    k = model.k;
    
    if(model.joint)
        a_dim = m+d+1;
    else
        a_dim = m;
    end
  
    P = reshape(theta(1:m*d),m,d);
    switch(method)
        case 'GL'
            
            GAMMA = theta(m*d+1);
            lnPHI = -0.5*Dxy(X,P)*GAMMA^2;
        case 'VL'
            
            GAMMA = theta(m*d+1:m*d+m)';
            lnPHI = -0.5*bsxfun(@times,Dxy(X,P),GAMMA.^2);
        case 'GD'
            
            GAMMA = theta(m*d+1:m*d+d)';
            
            lnPHI = zeros(n,m);
            for j = 1:m
                Delta = bsxfun(@minus,X,P(j, :));
                lnPHI(:,j) = -0.5*sum(power(bsxfun(@times,Delta,GAMMA),2),2);
            end
        case 'VD'

            GAMMA = reshape(theta(m*d+1:m*d+m*d),m,d);
            lnPHI = zeros(n,m);
            for j = 1:m
                Delta = bsxfun(@minus,X,P(j, :));
                lnPHI(:,j) = -0.5*sum(power(bsxfun(@times,Delta,GAMMA(j,:)),2),2);
            end
            
        case 'GC'
            
            GAMMA = reshape(theta(m*d+1:m*d+d*d),d,d);
            
            lnPHI = zeros(n,m);
            for j = 1:m
                Delta = bsxfun(@minus,X,P(j, :));
                lnPHI(:,j) = -0.5*sum(power(Delta*GAMMA',2),2);
            end
        case 'VC'
            
            GAMMA = reshape(theta(m*d+1:m*d+d*d*m),d,d,m);
            
            lnPHI = zeros(n,m);
            for j = 1:m
                Delta = bsxfun(@minus,X,P(j, :));
                lnPHI(:,j) = -0.5*sum(power(Delta*GAMMA(:,:,j)',2),2);
            end
            
    end
   
    g_dim = length(GAMMA(:));
    
    b = theta(m*d+g_dim+a_dim*k+1:m*d+g_dim+a_dim*k+k)';
    
    lnBeta = bsxfun(@plus,log(omega),repmat(b,n,1));
    
    PHI = exp(lnPHI);
    
    if(model.heteroscedastic)
        u = reshape(theta(m*d+g_dim+a_dim*k+k+1:m*d+g_dim+a_dim*k+k+m*k),m,k);
        lnBeta = PHI*u+lnBeta;
    end

    if(model.joint)
        PHI = [PHI X ones(n,1)];
    end
    
end