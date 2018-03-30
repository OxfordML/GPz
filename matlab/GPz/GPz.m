function [nlogML,grad,w,iSigma_w,PHI] = GPz(theta,model,X,Y,Psi,omega,training,validation)

global trainRMSE
global trainLL

global validRMSE
global validLL

k = model.k;
m = model.m;
method = model.method;
heteroscedastic = model.heteroscedastic;

[n,d] = size(X);

if(isempty(training))
    training = true(n,1);
end

if(isempty(omega))
    omega = ones(n,1);
end

n = sum(training);

g_dim = model.g_dim;
    
P = reshape(theta(1:m*d),m,d);

[PHI,Gamma,lnBeta_i] = getPHI(X,Psi,theta,model,training);

lnAlpha = reshape(theta(m*d+g_dim+1:m*d+g_dim+m*k),m,k);

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
w = zeros(m,k);
iSigma_w = zeros(m,m,k);
logdet = zeros(1,k);
dwda = zeros(m,k);
dlnPHI = zeros(n,m);
dlnAlpha = zeros(m,k);

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
    
    v = reshape(theta(m*d+g_dim+m*k+k+1:m*d+g_dim+m*k+k+m*k),m,k);
    
    lnTau = reshape(theta(m*d+g_dim+m*k+k+m*k+1:m*d+g_dim+m*k+k+m*k+m*k),m,k);
    tau = exp(lnTau);
    
    nlogML = nlogML-0.5*sum((v.^2).*tau)+0.5*sum(lnTau)-0.5*m*k*log(2*pi);
    dv = (PHI(:,1:m)'*dbeta)-v.*tau;
    dlnTau = -0.5*tau.*v.^2+0.5;
    dlnPHI = dlnPHI+(dbeta*v');    

end

nlogML = sum(nlogML)-0.5*log(2*pi)*sum(sum(omega(training,:)));


dPHI = dlnPHI.*PHI(:,1:m);

dP = zeros(size(P));
dGamma = zeros(size(Gamma));

missing = isnan(X(training,:));
    
list = true(n,1);
groups = logical([]);

while(sum(list)>0)
    first = find(list,1);
    group = false(n,1);
    group(list) = sum(abs(bsxfun(@minus,missing(list,:),missing(first,:))),2)==0;
    groups = [groups group];
    list(group)=false;
end

list = find(training);

for i=1:size(groups,2)
    
    for j=1:m

        group = groups(:,i);

        u = missing(find(group,1),:);
        o = ~u;
        
        Delta = bsxfun(@minus,X(training,o),P(j,o));

        if(method(2)=='C')
            
            iSigma = Gamma(:,:,j)'*Gamma(:,:,j);
            Sigma  = inv(iSigma);
            
            
            if(isempty(Psi))
                iSoo = inv(Sigma(o,o));
                dP(j,o) = dP(j,o)+(dPHI(group,j)'*Delta(group,:))*iSoo;

                diSoo = -0.5*bsxfun(@times,Delta(group,:),dPHI(group,j))'*Delta(group,:);

                GuuGuo = inv(iSigma(u,u))*iSigma(u,o);
                dGo = 2*(Gamma(:,o,j)-Gamma(:,u,j)*GuuGuo)*diSoo;
                dGamma(:,o,j) = dGamma(:,o,j)+dGo;
                dGamma(:,u,j) = dGamma(:,u,j)-dGo*GuuGuo';


%                 dCoo = 0.5*iSoo*(bsxfun(@times,Delta(group,o),dPHI(group,j))'*Delta(group,o))*iSoo;
%                 dLambda(:,o,j) = dLambda(:,o,j)+2*Lambda(:,o,j)*dCoo;
            else

                index = find(group);
                
                for id=1:sum(group)
                    
                    iPSoo = inv(Sigma(o,o)+Psi(o,o,list(index(id))));

                    dP(j,o) = dP(j,o)+dPHI(index(id),j)*Delta(index(id),:)*iPSoo;
                    
                    dSoo = 0.5*(inv(Sigma(o,o))-iPSoo+iPSoo*(Delta(index(id),:)'*Delta(index(id),:))*iPSoo);

                    diSoo = -Sigma(o,o)*dSoo*Sigma(o,o);

                    GuuGuo = inv(iSigma(u,u))*iSigma(u,o);
                    dGo = 2*(Gamma(:,o,j)-Gamma(:,u,j)*GuuGuo)*diSoo;
                    dGamma(:,o,j) = dGamma(:,o,j)+dPHI(index(id),j)*dGo;
                    dGamma(:,u,j) = dGamma(:,u,j)-dPHI(index(id),j)*dGo*GuuGuo';

%                     dLambda(:,o,j) = dLambda(:,o,j)+2*dPHI(index(id),j)*Lambda(:,o,j)*dSoo;
                end
                
            end

        else
            Sigma = Gamma(j,o).^-2;
            if(isempty(Psi))
                
                    dP(j,o) = dP(j,o)+(dPHI(group,j)'*Delta(group,:))./Sigma;

                    dGamma(j,o) = dGamma(j,o)-Gamma(j,o).*sum(bsxfun(@times,Delta(group,:).^2,dPHI(group,j)));

%                   dLambda(j,o) = dLambda(j,o)+Lambda(j,o).*sum(bsxfun(@times,Delta(group,:).^2,dPHI(group,j)))./Sigma(j,o).^2; % Variance

            else

                    Psi_plus_Sigma = bsxfun(@plus,Psi(list(group),o),Sigma);
                    
                    dP(j,o) = dP(j,o)+dPHI(group,j)'*(Delta(group,:)./Psi_plus_Sigma);

                    Psi_x_iSigma = (1+bsxfun(@rdivide,Psi(list(group),o),Sigma)).^-1;

                    dGamma(j,o) = dGamma(j,o)-Gamma(j,o).*(dPHI(group,j)'*(Delta(group,:).*Psi_x_iSigma).^2-dPHI(group,j)'*(bsxfun(@minus,bsxfun(@times,Psi_x_iSigma,Sigma),Sigma)));

%                   dLambda(j,:) = dLambda(j,:)+Lambda(j,:).*(dPHI(:,j)'*(power(Delta./Psi_plus_Sigma,2)-Psi_plus_Sigma.^-1)+sum(dPHI(:,j))*iSigma); % Variance

            end
        end
    end
end

switch(method)
                        
    case 'GL'
        dGamma = sum(dGamma(:));
    case 'VL'
        dGamma = sum(dGamma,2);
    case 'GD'
        dGamma = sum(dGamma,1);            
    case 'GC'
        dGamma = sum(dGamma,3);
end

grad = [dP(:);dGamma(:);dlnAlpha(:);db(:)];

if(heteroscedastic)
    grad = [grad;dv(:);dlnTau(:)];
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
