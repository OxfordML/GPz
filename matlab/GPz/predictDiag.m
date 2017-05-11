function [nu,Vg,gamma] = predictDiag(X,Psi,model,set,PHI,mu,lnBeta_i)
    
    m = model.m;
    k = model.k;
    d = model.d;

    method = model.method;
    joint = model.joint;
    
    theta = set.theta;
    w = set.w;
    iSigma_w = set.iSigma_w;
    
    switch(method)
                        
        case 'GL'
            Lambda = theta(m*d+1);
        case 'VL'
            Lambda = theta(m*d+1:m*d+m)';
        case 'GD'
            Lambda = theta(m*d+1:m*d+d)';
        case 'VD'
            Lambda = reshape(theta(m*d+1:m*d+m*d),m,d);
        case 'GC'
            Lambda = reshape(theta(m*d+1:m*d+d*d),d,d);
        case 'VC'
            Lambda = reshape(theta(m*d+1:m*d+d*d*m),d,d,m);
    end

    if(model.joint)
        a_dim = m+d+1;
    else
        a_dim = m;
    end
    
    l_dim = length(Lambda(:));
    
    if(model.heteroscedastic)
        v = reshape(theta(m*d+l_dim+a_dim*k+k+1:m*d+l_dim+a_dim*k+k+m*k),m,k);
    else
        v = zeros(m,k);
    end
    
    b = theta(m*d+l_dim+a_dim*k+1:m*d+l_dim+a_dim*k+k)';
    
    [n,k] = size(mu);

    gamma = zeros(n,k);
    nu = zeros(n,k);

    Ef2 = zeros(n,k);
    Vg = zeros(n,k);
    
    P = reshape(theta(1:m*d),m,d);
    
    for o=1:k
        
        switch(method)

            case 'GL'

                Lambda = repmat(theta(m*d+1),m,d);

            case 'VL'

                Lambda = repmat(theta(m*d+1:m*d+m),1,d);

            case 'GD'

                Lambda = repmat(theta(m*d+1:m*d+d)',m,1);

            case 'VD'

                Lambda = reshape(theta(m*d+1:m*d+m*d),m,d);
        end

        for i=1:m
            Si = Lambda(i,:).^-2;
            zi = sqrt(exp(sum(log(Si))));

            for j=1:i

                Sj = Lambda(j,:).^-2;
                zj = sqrt(exp(sum(log(Sj))));

                Cij = (Si.^-1+Sj.^-1).^-1;
                cij = (P(i,:)./Si+P(j,:)./Sj).*Cij;

                Zij = zi*zj*exp(-0.5*sum(power(P(i,:)-P(j,:),2)./(Si+Sj))-0.5*sum(log(Si+Sj)));

                Delta = bsxfun(@minus,X,cij);

                Cij_plus_Psi = bsxfun(@plus,Psi,Cij);

                Nxc = exp(-0.5*sum((Delta.^2)./Cij_plus_Psi,2)-0.5*sum(log(Cij_plus_Psi),2));

                Ef2(:,o) = Ef2(:,o)+2*w(i,o)*w(j,o)*Zij*Nxc;
                Vg(:,o)  = Vg(:,o)+2*v(i,o)*v(j,o)*Zij*Nxc;
                nu(:,o)  = nu(:,o)+2*iSigma_w(i,j,o)*Zij*Nxc;

            end

            Ef2(:,o) = Ef2(:,o)-w(i,o)*w(j,o)*Zij*Nxc;
            Vg(:,o)  = Vg(:,o)-v(i,o)*v(j,o)*Zij*Nxc;
            nu(:,o)  = nu(:,o)-iSigma_w(i,j,o)*Zij*Nxc;

            if(joint)

                cij = bsxfun(@plus,bsxfun(@times,X,Si),bsxfun(@times,Psi,P(i,:)))./bsxfun(@plus,Psi,Si);

                for j=1:d

                    Nxc = PHI(:,i).*cij(:,j);

                    Ef2(:,o) = Ef2(:,o)+2*w(i,o)*w(m+j,o)*Nxc;
                    nu(:,o)  = nu(:,o)+2*iSigma_w(i,m+j,o)*Nxc;

                end

                Ef2(:,o) = Ef2(:,o)+2*w(i,o)*w(m+d+1,o)*PHI(:,i);
                nu(:,o)  = nu(:,o)+2*iSigma_w(i,m+d+1,o)*PHI(:,i);
            end
        end

        if(joint)

            for i=1:d
                for j=1:i-1
                    Exx = X(:,i).*X(:,j);

                    Ef2(:,o) = Ef2(:,o)+2*w(m+i,o)*w(m+j,o)*Exx;
                    nu(:,o)  = nu(:,o)+2*iSigma_w(m+i,m+j,o)*Exx;
                end

                Ef2(:,o) = Ef2(:,o)+2*w(m+i,o)*w(m+d+1,o)*X(:,i);
                nu(:,o)  = nu(:,o)+2*iSigma_w(m+i,m+d+1,o)*X(:,i);
                
                Exx = Psi(:,i)+X(:,i).^2;
                Ef2(:,o) = Ef2(:,o)+Exx*w(m+i,o)^2;
                nu(:,o)  = nu(:,o)+Exx*iSigma_w(m+i,m+i,o);
            end


            Ef2(:,o) = Ef2(:,o)+w(m+d+1,o)^2;
            nu(:,o)  = nu(:,o)+iSigma_w(m+d+1,m+d+1,o);
        end

        Vg(:,o) = Vg(:,o)-(lnBeta_i(:,o)-b(o)).^2;
        gamma(:,o) = Ef2(:,o)-mu(:,o).^2;

    end
    
end