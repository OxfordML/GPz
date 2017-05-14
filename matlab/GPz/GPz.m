function [nlogML,grad,w,iSigma_w,PHI] = GPz(theta,model,X,Y,Psi,omega,training,validation)

global trainRMSE
global trainLL

global validRMSE
global validLL

k = model.k;
m = model.m;
method = model.method;
heteroscedastic = model.heteroscedastic;
joint = model.joint;

[n,d] = size(X);

learnPsi = ischar(Psi);
if(learnPsi)
    if(method(2)=='C')
        S = reshape(theta(end-d*d+1:end),d,d);
        Psi = reshape(repmat(S'*S,1,n),d,d,n);
    else
        S = reshape(theta(end-d+1:end),1,d);
        Psi = ones(n,1)*S.^2;
    end
    dS = zeros(size(S));
end

if(isempty(training))
    training = true(n,1);
end

if(isempty(omega))
    omega = ones(n,1);
end

n = sum(training);

if(joint)
    a_dim = m+d+1;
else
    a_dim = m;
end

P = reshape(theta(1:m*d),m,d);

[PHI,Lambda,lnBeta_i] = getPHI(X,Psi,theta,model,training);

l_dim = length(Lambda(:));

lnAlpha = reshape(theta(m*d+l_dim+1:m*d+l_dim+a_dim*k),a_dim,k);

if(isempty(Y))
    nlogML = 0;
    grad = 0;
    w = 0;
    iSigma_w = 0;
    return
end


beta = exp(-lnBeta_i);
beta_i = beta.^-1;
df = -beta;


omega_x_beta = bsxfun(@times,beta,omega(training,:));

alpha = exp(lnAlpha);
da = alpha;

nu = zeros(n,k);
w = zeros(a_dim,k);
iSigma_w = zeros(a_dim,a_dim,k);
logdet = zeros(1,k);
dwda = zeros(a_dim,k);
dlnPHI = zeros(n,m);
dlnAlpha = zeros(a_dim,k);

for i=1:k
    
    BxPHI = bsxfun(@times,PHI,omega_x_beta(:,i));

    SIGMA = BxPHI'*PHI+diag(alpha(:,i));
    
    [iSigma_w(:,:,i),logdet(i)] = inv_logdet(SIGMA);
    
    nu(:,i) = sum(PHI.*(PHI*iSigma_w(:,:,i)),2);
    w(:,i) = iSigma_w(:,:,i)*BxPHI'*Y(training,i);
    dwda(:,i) = -iSigma_w(:,:,i)*(da(:,i).*w(:,i));
    dlnPHI = dlnPHI-(BxPHI*iSigma_w(:,1:m,i));
    dlnAlpha(:,i) = -0.5*diag(iSigma_w(:,:,i)).*da(:,i);

end

delta = PHI*w-Y(training,:);

omega_beta_x_delta = omega_x_beta.*delta;

nlogML = -0.5*sum(omega_beta_x_delta.*delta)-0.5*sum(alpha.*w.^2)+0.5*sum(lnAlpha)-0.5*logdet;
nlogML = nlogML+0.5*sum(bsxfun(@times,-lnBeta_i,omega(training,:)));

if(nargout>2)
    grad = 0;
    return
end

dlnAlpha = dlnAlpha-(PHI'*omega_beta_x_delta).*dwda-alpha.*w.*dwda-0.5*da.*w.^2+0.5;
dlnPHI = dlnPHI-(omega_beta_x_delta*w(1:m,:)');


dbeta = bsxfun(@times,0.5*df.*(beta_i-(delta.^2+nu)),omega(training,:));
db = sum(dbeta);

if(heteroscedastic)
    
    v = reshape(theta(m*d+l_dim+a_dim*k+k+1:m*d+l_dim+a_dim*k+k+m*k),m,k);
    
    lnTau = reshape(theta(m*d+l_dim+a_dim*k+k+m*k+1:m*d+l_dim+a_dim*k+k+m*k+m*k),m,k);
    tau = exp(lnTau);
    
    nlogML = nlogML-0.5*sum((v.^2).*tau)+0.5*sum(lnTau)-0.5*m*k*log(2*pi);
    dv = (PHI(:,1:m)'*dbeta)-v.*tau;
    dlnTau = -0.5*tau.*v.^2+0.5;
    dlnPHI = dlnPHI+(dbeta*v');    
    

end

nlogML = sum(nlogML)-0.5*log(2*pi)*sum(sum(omega(training,:)));


dPHI = dlnPHI.*PHI(:,1:m);

dP = zeros(size(P));
dLambda = zeros(size(Lambda));

list = find(training);

for j=1:m
    
    Delta = bsxfun(@minus,X(training,:),P(j,:));
    
    switch(method)
        case 'GL'
            
            if(j==1)
                Sigma = Lambda.^-2;
                iSigma = Lambda.^2;
            end
            
            if(isempty(Psi))
                
                dP(j,:) = dP(j,:)+(dPHI(:,j)'*Delta)*iSigma;
                dLambda = dLambda-Lambda*sum(sum(bsxfun(@times,Delta.^2,dPHI(:,j))));
%                 dLambda = dLambda+Lambda*sum(sum(bsxfun(@times,Delta.^2,dPHI(:,j))))*iSigma^2; % Variance
                                        
            else

               
                Psi_plus_Sigma = Psi(training,:)+Sigma;
                Psi_x_iSigma = (1+Psi(training,:)*iSigma).^-1;

                dP(j,:) = dP(j,:)+dPHI(:,j)'*bsxfun(@rdivide,Delta,Psi_plus_Sigma);

                dLambda = dLambda-Lambda*sum((dPHI(:,j)'*(Delta.*Psi_x_iSigma).^2-dPHI(:,j)'*(bsxfun(@minus,bsxfun(@times,Psi_x_iSigma,Sigma),Sigma))));
%                 dLambda = dLambda+Lambda*sum((dPHI(:,j)'*(Delta.*Psi_x_iSigma).^2-dPHI(:,j)'*(bsxfun(@minus,bsxfun(@times,Psi_x_iSigma,Sigma),Sigma))))*iSigma^2; % Variance
  
                
            end
        case 'VL'
            
            Sigma = Lambda(j).^-2;
            iSigma = Lambda(j).^2;
            
            if(isempty(Psi))
                
                dP(j,:) = dP(j,:)+(dPHI(:,j)'*Delta)*iSigma;
                dLambda(j) = dLambda(j)-Lambda(j)*sum(sum(bsxfun(@times,Delta.^2,dPHI(:,j))));
%                 dLambda(j) = dLambda(j)+Lambda(j)*sum(sum(bsxfun(@times,Delta.^2,dPHI(:,j))))*iSigma^2; % Variance
                                        
            else

                Psi_plus_Sigma = Psi(training,:)+Sigma;
                Psi_x_iSigma = (1+Psi(training,:)*iSigma).^-1;

                dP(j,:) = dP(j,:)+dPHI(:,j)'*bsxfun(@rdivide,Delta,Psi_plus_Sigma);

                dLambda(j) = dLambda(j)-Lambda(j)*sum((dPHI(:,j)'*(Delta.*Psi_x_iSigma).^2-dPHI(:,j)'*(bsxfun(@minus,bsxfun(@times,Psi_x_iSigma,Sigma),Sigma))));
%                 dLambda(j) = dLambda(j)+Lambda(j)*sum((dPHI(:,j)'*(Delta.*Psi_x_iSigma).^2-dPHI(:,j)'*(bsxfun(@minus,bsxfun(@times,Psi_x_iSigma,Sigma),Sigma))))*iSigma^2; % Variance
  
                
            end
        case 'GD'
            
            if(j==1)
                Sigma = Lambda.^-2;
                iSigma = Lambda.^2;
            end
            
            if(isempty(Psi))
                
                    dP(j,:) = dP(j,:)+(dPHI(:,j)'*Delta).*iSigma;
                    
                    dLambda = dLambda-Lambda.*sum(bsxfun(@times,Delta.^2,dPHI(:,j)));
%                     dLambda = dLambda+Lambda.*sum(bsxfun(@times,Delta.^2,dPHI(:,j))).*iSigma.^2; % Variance
                                        
            else

                
                Psi_plus_Sigma = bsxfun(@plus,Psi(training,:),Sigma);

                dP(j,:) = dP(j,:)+dPHI(:,j)'*(Delta./Psi_plus_Sigma);
                
                Psi_x_iSigma = (1+bsxfun(@times,Psi(training,:),iSigma)).^-1;
                dLambda = dLambda-Lambda.*(dPHI(:,j)'*(Delta.*Psi_x_iSigma).^2-dPHI(:,j)'*(bsxfun(@minus,bsxfun(@times,Psi_x_iSigma,Sigma),Sigma)));

%                 dLambda = dLambda+Lambda.*(dPHI(:,j)'*(power(Delta./Psi_plus_Sigma,2)-Psi_plus_Sigma.^-1)+sum(dPHI(:,j))*iSigma); % Variance
  
                
            end
        case 'VD'

            Sigma = Lambda(j,:).^-2;
            iSigma = Lambda(j,:).^2;
            
            if(isempty(Psi))
                
                    
                dP(j,:) = dP(j,:)+(dPHI(:,j)'*Delta).*iSigma;

                dLambda(j,:) = dLambda(j,:)-Lambda(j,:).*sum(bsxfun(@times,Delta.^2,dPHI(:,j)));

%                 dLambda(j,:) = dLambda(j,:)+Lambda(j,:).*sum(bsxfun(@times,Delta.^2,dPHI(:,j))).*iSigma.^2; % Variance
                                        
            else


                Psi_plus_Sigma = bsxfun(@plus,Psi(training,:),Sigma);

                dP(j,:) = dP(j,:)+dPHI(:,j)'*(Delta./Psi_plus_Sigma);

                Psi_x_iSigma = (1+bsxfun(@times,Psi(training,:),iSigma)).^-1;
                dLambda(j,:) = dLambda(j,:)-Lambda(j,:).*(dPHI(:,j)'*(Delta.*Psi_x_iSigma).^2-dPHI(:,j)'*(bsxfun(@minus,bsxfun(@times,Psi_x_iSigma,Sigma),Sigma)));
                
                if(learnPsi)
                    dS = dS+S.*(dPHI(:,j)'*(power(Delta./Psi_plus_Sigma,2)-Psi_plus_Sigma.^-1));
                end
               
%                 dLambda(j,:) = dLambda(j,:)+Lambda(j,:).*(dPHI(:,j)'*(power(Delta./Psi_plus_Sigma,2)-Psi_plus_Sigma.^-1)+sum(dPHI(:,j))*iSigma); % Variance
  

            end 
        case 'GC'

            if(j==1)
                iSigma = Lambda'*Lambda;
                Sigma = inv(iSigma);
            end
            
            if(isempty(Psi))
                
                
                dP(j,:) = dP(j,:)+(dPHI(:,j)'*Delta)*iSigma;

                diSigma = -0.5*bsxfun(@times,Delta,dPHI(:,j))'*Delta;
                
%                 dSigma = -iSigma*diSigma*iSigma;
%                 dLambda = dLambda+2*Lambda*dSigma; % Variance
                
                dLambda = dLambda+2*Lambda*diSigma;
                
                
            else


                for i=1:n

                    invCS = inv(Sigma+Psi(:,:,list(i)));
                    
                    dP(j,:) = dP(j,:)+dPHI(i,j)*Delta(i,:)*invCS;
            
                    dSigma = 0.5*(iSigma-invCS+invCS*(Delta(i,:)'*Delta(i,:))*invCS);
                    
                    diSigma = -Sigma*dSigma*Sigma;
                    dLambda = dLambda+2*dPHI(i,j)*Lambda*diSigma;                

%                     dLambda = dLambda+2*dPHI(i,j)*Lambda*dSigma; % Variance
                    
                end
            end
        case 'VC'

            iSigma = Lambda(:,:,j)'*Lambda(:,:,j);
            Sigma = inv(iSigma);
            
            if(isempty(Psi))
                
                dP(j,:) = dP(j,:)+(dPHI(:,j)'*Delta)*iSigma;

                diSigma = -0.5*bsxfun(@times,Delta,dPHI(:,j))'*Delta;
                
                
                dLambda(:,:,j) = 2*Lambda(:,:,j)*diSigma;
                

%                 dSigma = -iSigma*diSigma*iSigma;
%                 dLambda(:,:,j) = 2*Lambda(:,:,j)*dSigma; % Variance
                
            else
                
                for i=1:n

                    invCS = inv(Sigma+Psi(:,:,list(i)));
                    
                    dP(j,:) = dP(j,:)+dPHI(i,j)*Delta(i,:)*invCS;
            
                    dSigma = 0.5*(iSigma-invCS+invCS*(Delta(i,:)'*Delta(i,:))*invCS);
                    
                    diSigma = -Sigma*dSigma*Sigma;
                    dLambda(:,:,j) = dLambda(:,:,j)+2*dPHI(i,j)*Lambda(:,:,j)*diSigma;                

%                     dLambda(:,:,j) = dLambda(:,:,j)+2*dPHI(i,j)*Lambda(:,:,j)*dSigma; % Variance
                    
                end
            end
    end
    
end

grad = [dP(:);dLambda(:);dlnAlpha(:);db(:)];

if(heteroscedastic)
    grad = [grad;dv(:);dlnTau(:)];
end

if(learnPsi)
    grad = [grad;dS(:)];
end

nlogML = -nlogML/(n*k);
grad = -grad/(n*k);

trainRMSE = sqrt(sum(sum(bsxfun(@times,delta.^2,omega(training))))/(n*k));
trainLL = sum(sum(bsxfun(@times,-0.5*beta.*delta.^2+0.5*log(beta),omega(training,:))))/(n*k)-0.5*log(2*pi);

if(~isempty(validation))
    
    n = sum(validation);
    
    [PHI,~,lnBeta_i] = getPHI(X,Psi,theta,model,validation);
    
    beta = exp(-lnBeta_i);


    nu = zeros(n,k);
    
    for i=1:k
        nu(:,i) = sum(PHI.*(PHI*iSigma_w(:,:,i)),2);
    end
    
    pred = PHI*w;
    delta = pred-Y(validation,:);
    
    
    validRMSE = sqrt(sum(sum(bsxfun(@times,delta.^2,omega(validation))))/(n*k));
    validLL = sum(sum(bsxfun(@times,-0.5*beta.*delta.^2+0.5*log(beta),omega(validation,:))))/(n*k)-0.5*log(2*pi);
    
end

end
