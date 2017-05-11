function [nu,Vg,gamma] = predictCov(X,Psi,model,set,PHI,mu,lnBeta_i)
    
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
    
    for id=1:n
        
        for o=1:k

            switch(method)

                case 'GC'

                    Lambda = reshape(repmat(reshape(theta(m*d+1:m*d+d*d),d,d),1,m),d,d,m);

                case 'VC'

                    Lambda = reshape(theta(m*d+1:m*d+d*d*m),d,d,m);
            end

            for i=1:m
                
                Gi = Lambda(:,:,i)'*Lambda(:,:,i);
                Si = inv(Lambda(:,:,i)'*Lambda(:,:,i));
                
                zi = sqrt(exp(sum(log(svd(Si)))));

                for j=1:i
                    

                    Gj = Lambda(:,:,j)'*Lambda(:,:,j);
                    Sj = inv(Lambda(:,:,j)'*Lambda(:,:,j));
                    zj = sqrt(exp(sum(log(svd(Sj)))));

                    Cij = inv(Gi+Gj);
                    cij = (P(i,:)*Gi+P(j,:)*Gj)*Cij;
                    
                    Delta = P(i,:)-P(j,:);

                    Zij = zi*zj*exp(-0.5*(Delta/(Si+Sj))*Delta'-0.5*sum(log(svd(Si+Sj))));

                    Delta = X(id,:)-cij;

                    Cij_plus_Psi = Psi(:,:,id)+Cij;

                    Nxc = exp(-0.5*(Delta/Cij_plus_Psi)*Delta'-0.5*sum(log(svd(Cij_plus_Psi))));

                    Ef2(id,o) = Ef2(id,o)+2*w(i,o)*w(j,o)*Zij*Nxc;
                    Vg(id,o)  = Vg(id,o)+2*v(i,o)*v(j,o)*Zij*Nxc;
                    nu(id,o)  = nu(id,o)+2*iSigma_w(i,j,o)*Zij*Nxc;

                end

                Ef2(id,o) = Ef2(id,o)-w(i,o)*w(j,o)*Zij*Nxc;
                Vg(id,o)  = Vg(id,o)-v(i,o)*v(j,o)*Zij*Nxc;
                nu(id,o)  = nu(id,o)-iSigma_w(i,j,o)*Zij*Nxc;

                if(joint)

                    cij = (X(id,:)+(P(i,:)*Gi)*Psi(:,:,id))*inv(Psi(:,:,id)+Si)*Si;

                    for j=1:d

                        Nxc = PHI(id,i)*cij(j);

                        Ef2(id,o) = Ef2(id,o)+2*w(i,o)*w(m+j,o)*Nxc;
                        nu(id,o)  = nu(id,o)+2*iSigma_w(i,m+j,o)*Nxc;

                    end

                    Ef2(id,o) = Ef2(id,o)+2*w(i,o)*w(m+d+1,o)*PHI(id,i);
                    nu(id,o)  = nu(id,o)+2*iSigma_w(i,m+d+1,o)*PHI(id,i);
                end
            end

            if(joint)

                Exx = Psi(:,:,id)+X(id,:)'*X(id,:);
                for i=1:d
                    for j=1:i-1
                        
                        Ef2(id,o) = Ef2(id,o)+2*w(m+i,o)*w(m+j,o)*Exx(i,j);
                        nu(id,o)  = nu(id,o)+2*iSigma_w(m+i,m+j,o)*Exx(i,j);
                    end

                    Ef2(id,o) = Ef2(id,o)+2*w(m+i,o)*w(m+d+1,o)*X(id,i);
                    nu(id,o)  = nu(id,o)+2*iSigma_w(m+i,m+d+1,o)*X(id,i);
                    
                    Ef2(id,o) = Ef2(id,o)+Exx(i,i)*w(m+i,o)^2;
                    nu(id,o)  = nu(id,o)+Exx(i,i)*iSigma_w(m+i,m+i,o);

                end

                Ef2(id,o) = Ef2(id,o)+w(m+d+1,o)^2;
                nu(id,o)  = nu(id,o)+iSigma_w(m+d+1,m+d+1,o);
            end

            Vg(id,o) = Vg(id,o)-(lnBeta_i(id,o)-b(o)).^2;
            gamma(id,o) = Ef2(id,o)-mu(id,o).^2;

        end
    end
    
end