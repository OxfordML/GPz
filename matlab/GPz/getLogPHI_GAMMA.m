function [lnPHI,GAMMA,g_dim] = getLogPHI_GAMMA(X,theta,method,m)

    [n,d] = size(X);
    P = reshape(theta(1:m*d),m,d);
    switch(method)
        case 'GL'
            g_dim = 1;
            GAMMA = theta(m*d+1);
            lnPHI = -0.5*Dxy(X,P)*GAMMA^2;
        case 'VL'
            g_dim = m;
            GAMMA = theta(m*d+1:m*d+m)';
            lnPHI = -0.5*bsxfun(@times,Dxy(X,P),GAMMA.^2);
        case 'GD'
            g_dim = d;
            GAMMA = theta(m*d+1:m*d+d)';
            
            lnPHI = zeros(n,m);
            for j = 1:m
                Delta = bsxfun(@minus,X,P(j, :));
                lnPHI(:,j) = -0.5*sum(power(bsxfun(@times,Delta,GAMMA),2),2);
            end
        case 'VD'
            g_dim = d*m;

            GAMMA = reshape(theta(m*d+1:m*d+m*d),m,d);
            lnPHI = zeros(n,m);
            for j = 1:m
                Delta = bsxfun(@minus,X,P(j, :));
                lnPHI(:,j) = -0.5*sum(power(bsxfun(@times,Delta,GAMMA(j,:)),2),2);
            end
            
        case 'GC'
            g_dim = d*d;
            GAMMA = reshape(theta(m*d+1:m*d+d*d),d,d);
            
            lnPHI = zeros(n,m);
            for j = 1:m
                Delta = bsxfun(@minus,X,P(j, :));
                lnPHI(:,j) = -0.5*sum(power(Delta*GAMMA',2),2);
            end
        case 'VC'
            
            g_dim = d*d*m;
            GAMMA = reshape(theta(m*d+1:m*d+d*d*m),d,d,m);
            
            lnPHI = zeros(n,m);
            for j = 1:m
                Delta = bsxfun(@minus,X,P(j, :));
                lnPHI(:,j) = -0.5*sum(power(Delta*GAMMA(:,:,j)',2),2);
            end
            
    end
end