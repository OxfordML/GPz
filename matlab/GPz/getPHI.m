function [PHI,Lambda,lnBeta_i,N] = getPHI(X,Psi,theta,model,set)

    if(isempty(set))
        set = true(size(X,1),1);
    end

    n = sum(set);
    d = model.d;
    m = model.m;
    k = model.k;

    method = model.method;
    
    if(model.joint)
        a_dim = m+d+1;
    else
        a_dim = m;
    end
     
    P = reshape(theta(1:m*d),m,d);
    
    list = find(set);
    
    switch(method)
        case 'GL'
            
            Lambda = theta(m*d+1);
            lnPHI = zeros(n,m);
            lnN = zeros(n,m);
            
            Ci = Lambda^2;
            C = Lambda^-2;

            for j = 1:m

                Delta = bsxfun(@minus,X(set,:),P(j, :));
                
                if(isempty(Psi))
                    lnPHI(:,j) = -0.5*sum(Ci*Delta.^2,2);
                    lnN(:,j) = lnPHI(:,j)-0.5*d*log(C)-0.5*d*log(2*pi);
                else

                    SxPlusC = Psi(set,:)+C;

                    lnPHI(:,j) = -0.5*sum((Delta.^2)./SxPlusC,2)-0.5*sum(log(1+Psi(set,:)*Ci),2);
                    lnN(:,j) = lnPHI(:,j)-0.5*d*log(C)-0.5*d*log(2*pi);
                end

            end
        case 'VL'
            
            Lambda = theta(m*d+1:m*d+m)';
            lnPHI = zeros(n,m);
            lnN = zeros(n,m);
            
            for j = 1:m

                Ci = Lambda(j)^2;
                C = Lambda(j)^-2;
                
                Delta = bsxfun(@minus,X(set,:),P(j, :));
                
                if(isempty(Psi))
                    
                    lnPHI(:,j) = -0.5*sum(Ci*Delta.^2,2);
                    lnN(:,j) = lnPHI(:,j)-0.5*d*log(C)-0.5*d*log(2*pi);
                else

                    SxPlusC = Psi(set,:)+C;

                    lnPHI(:,j) = -0.5*sum((Delta.^2)./SxPlusC,2)-0.5*sum(log(1+Psi(set,:)*Ci),2);
                    lnN(:,j) = lnPHI(:,j)-0.5*d*log(C)-0.5*d*log(2*pi);

                end

            end
        case 'GD'
            
            Lambda = theta(m*d+1:m*d+d)';
            
            lnPHI = zeros(n,m);
            lnN = zeros(n,m);
            Ci = Lambda.^2;
            C = Lambda.^-2;
            for j = 1:m

                Delta = bsxfun(@minus,X(set,:),P(j, :));
                
                if(isempty(Psi))
                    
                    lnPHI(:,j) = -0.5*sum((bsxfun(@times,Delta.^2,Ci)),2);
                    lnN(:,j) = lnPHI(:,j)-0.5*sum(log(C))-0.5*d*log(2*pi);
                    
                else

                    SxPlusC = bsxfun(@plus,Psi(set,:),C);

                    lnPHI(:,j) = -0.5*sum((Delta.^2)./SxPlusC,2)-0.5*sum(log(1+bsxfun(@times,Psi(set,:),Ci)),2);
                    lnN(:,j) = lnPHI(:,j)-0.5*sum(log(C))-0.5*d*log(2*pi);
                end

            end
        case 'VD'

            Lambda = reshape(theta(m*d+1:m*d+m*d),m,d);
            
            lnPHI = zeros(n,m);
            lnN = zeros(n,m);
            
            for j = 1:m

                Ci = Lambda(j,:).^2;
                C = Lambda(j,:).^-2;
                
                Delta = bsxfun(@minus,X(set,:),P(j, :));
                
                if(isempty(Psi))

                    lnPHI(:,j) = -0.5*sum((bsxfun(@times,Delta.^2,Ci)),2);
                    lnN(:,j) = lnPHI(:,j)-0.5*sum(log(C))-0.5*d*log(2*pi);

                else
                        
                    SxPlusC = bsxfun(@plus,Psi(set,:),C);

                    lnPHI(:,j) = -0.5*sum((Delta.^2)./SxPlusC,2)-0.5*sum(log(1+bsxfun(@times,Psi(set,:),Ci)),2);
                    lnN(:,j) = lnPHI(:,j)-0.5*sum(log(C))-0.5*d*log(2*pi);

                end

            end
            
        case 'GC'
            
            Lambda = reshape(theta(m*d+1:m*d+d*d),d,d);

            Ci = Lambda'*Lambda;
            C = inv(Ci);

            lnPHI = zeros(n,m);
            lnN = zeros(n,m);
            
            
            for j = 1:m
                
                Delta = bsxfun(@minus,X(set,:),P(j, :));
                
                if(isempty(Psi))
                        
                    lnPHI(:,j) = -0.5*sum((Delta/C).*Delta,2);
                    lnN(:,j) = lnPHI(:,j)-0.5*sum(log(svd(C)))-0.5*d*log(2*pi);
                    
                else

                    for i=1:n
                                                
                        lnPHI(i,j) = -0.5*sum((Delta(i,:)/(C+Psi(:,:,list(i)))).*Delta(i,:),2)-0.5*sum(log(svd(eye(d)+C\Psi(:,:,list(i)))));
                        lnN(i,j) = lnPHI(i,j)-0.5*sum(log(svd(C)))-0.5*d*log(2*pi);

                    end
                end
            end
        case 'VC'
            
            Lambda = reshape(theta(m*d+1:m*d+d*d*m),d,d,m);
            
            lnPHI = zeros(n,m);
            lnN = zeros(n,m);
            
            
            for j = 1:m
                
                Ci = Lambda(:,:,j)'*Lambda(:,:,j);
                C = inv(Ci);
                
                Delta = bsxfun(@minus,X(set,:),P(j, :));
                
                if(isempty(Psi))
                    lnPHI(:,j) = -0.5*sum((Delta/C).*Delta,2);
                    lnN(:,j) = lnPHI(:,j)-0.5*sum(log(svd(C)))-0.5*d*log(2*pi);
                else

                    for i=1:n
                        
                        lnPHI(i,j) = -0.5*sum((Delta(i,:)/(C+Psi(:,:,list(i)))).*Delta(i,:),2)+0.5*sum(log(svd(C)))-0.5*sum(log(svd(C+Psi(:,:,list(i)))));
                        lnN(i,j) = lnPHI(i,j)-0.5*sum(log(svd(C)))-0.5*d*log(2*pi);
                    end
                end

            end
    end
    
    PHI = exp(lnPHI);
    N = exp(lnN);
    l_dim = length(Lambda(:));
    b = theta(m*d+l_dim+a_dim*k+1:m*d+l_dim+a_dim*k+k)';
    
    lnBeta_i = repmat(b,n,1);

    if(model.heteroscedastic)
        v = reshape(theta(m*d+l_dim+a_dim*k+k+1:m*d+l_dim+a_dim*k+k+m*k),m,k);

        lnBeta_i = lnBeta_i+PHI*v;
    end
    
    if(model.joint)
        
        PHI = [PHI X(set,:) ones(n,1)];
    end
    
end