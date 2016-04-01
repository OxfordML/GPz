function [nlogML,grad,w,SIGMAi,PHI,lnBeta] = GPz(theta,model,X,Y,omega,training,validation)

global trainRMSE
global trainLL

global validRMSE
global validLL

k = model.k;
m = model.m;
method = model.method;
joint = model.joint;
heteroscedastic = model.heteroscedastic;

n = size(X,1);

if(isempty(training))
    training = true(n,1);
end

if(isempty(omega))
    omega = ones(n,1);
end

n = sum(training);

d = size(X,2);

if(joint)
    a_dim = m+d+1;
else
    a_dim = m;
end

P = reshape(theta(1:m*d),m,d);

[PHI,lnBeta,GAMMA] = getPHI(X(training,:),omega(training,:),theta,model);

g_dim = length(GAMMA(:));

lnAlpha = reshape(theta(m*d+g_dim+1:m*d+g_dim+a_dim*k),a_dim,k);

if(isempty(Y))
    nlogML = 0;
    grad = 0;
    w = 0;
    SIGMAi = 0;
    return
end

beta = exp(lnBeta);
alpha = exp(lnAlpha);

nu = zeros(n,k);
w = zeros(a_dim,k);
dwda = zeros(a_dim,k);
SIGMAi = zeros(a_dim,a_dim,k);
logdet = zeros(1,k);
dlnPHI = zeros(n,m);
dlnAlpha = zeros(a_dim,k);

for i=1:k
    
    BxPHI = bsxfun(@times,PHI,beta(:,i));

    SIGMA = BxPHI'*PHI+diag(alpha(:,i));

    [U,S,~] = svd(SIGMA);

    SIGMAi(:,:,i) = (U/S)*U';
    logdet(i) = sum(log(diag(S)));

    nu(:,i) = sum(PHI.*(PHI*SIGMAi(:,:,i)),2);

    w(:,i) = SIGMAi(:,:,i)*BxPHI'*Y(training,i);

    dwda(:,i) = -SIGMAi(:,:,i)*(alpha(:,i).*w(:,i));
    
    dlnPHI = dlnPHI-(BxPHI*SIGMAi(:,1:m,i)).*PHI(:,1:m);
    dlnAlpha(:,i) = -0.5*diag(SIGMAi(:,:,i)).*alpha(:,i);

end

delta = PHI*w-Y(training,:);

nlogML = -0.5*sum(beta.*delta.^2)+0.5*sum(lnBeta)-0.5*sum(alpha.*w.^2)+0.5*sum(lnAlpha)-0.5*logdet;

if(nargout>2)
    grad = 0;
    return
end

dlnAlpha = dlnAlpha-(PHI'*(beta.*delta)).*dwda-alpha.*w.*dwda-0.5*alpha.*w.^2+0.5;
dlnPHI= dlnPHI-((beta.*delta)*w(1:m,:)').*PHI(:,1:m);

dbeta = -0.5*(beta.*delta.^2+beta.*nu)+0.5;
db = sum(dbeta);

if(heteroscedastic)
    u = reshape(theta(m*d+g_dim+a_dim*k+k+1:m*d+g_dim+a_dim*k+k+m*k),m,k);
    lnEta = reshape(theta(m*d+g_dim+a_dim*k+k+m*k+1:m*d+g_dim+a_dim*k+k+m*k+m*k),m,k);
    eta = exp(lnEta);
    nlogML = nlogML-0.5*sum(u.^2.*eta)+0.5*sum(lnEta);
    du = PHI(:,1:m)'*dbeta-u.*eta;
    dlnEta = -0.5*eta.*u.^2+0.5;
    dlnPHI = dlnPHI+(dbeta*u').*PHI(:,1:m);    
end

dP = zeros(size(P));
dGAMMA = zeros(size(GAMMA));

for j=1:m
    Delta = bsxfun(@minus,X(training,:),P(j,:));
    switch(method)
        case 'GL'
            dP(j,:) = dlnPHI(:,j)'*Delta*GAMMA^2;
            dGAMMA = dGAMMA-sum(dlnPHI(:,j).*sum(Delta.^2,2))*GAMMA;
        case 'VL'
            dP(j,:) = dlnPHI(:,j)'*Delta*GAMMA(j)^2;
            dGAMMA(j) = -sum(dlnPHI(:,j).*sum(Delta.^2,2))*GAMMA(j);
        case 'GD'
            dP(j,:) = dlnPHI(:,j)'*bsxfun(@times,Delta,GAMMA.^2);
            dGAMMA = dGAMMA-sum(bsxfun(@times,bsxfun(@times,Delta,dlnPHI(:,j)),GAMMA).*Delta);
        case 'VD'
            dP(j,:) = dlnPHI(:,j)'*bsxfun(@times,Delta,GAMMA(j,:).^2);
            dGAMMA(j,:) = -sum(bsxfun(@times,bsxfun(@times,Delta,dlnPHI(:,j)),GAMMA(j,:)).*Delta);
        case 'GC'
            dP(j,:) = dlnPHI(:,j)'*Delta*(GAMMA'*GAMMA);
            dGj = -GAMMA*bsxfun(@times,Delta,dlnPHI(:,j))'*Delta;
            dGAMMA = dGAMMA+dGj;
        case 'VC'
            dP(j,:) = dlnPHI(:,j)'*Delta*(GAMMA(:,:,j)'*GAMMA(:,:,j));
            dGj = -GAMMA(:,:,j)*bsxfun(@times,Delta,dlnPHI(:,j))'*Delta;
            dGAMMA(:,:,j) = dGj;
    end
end

grad = [dP(:);dGAMMA(:);dlnAlpha(:);db(:)];

if(heteroscedastic)
    grad = [grad;du(:);dlnEta(:)];
end

nlogML = -sum(nlogML)/(n*k);
grad = -grad/(n*k);

sigma = nu+exp(-lnBeta);

trainRMSE = sqrt(sum(bsxfun(@times,delta.^2,omega(training)))/(n*k));
trainLL = sum(sum(-0.5*delta.^2./sigma-0.5*log(sigma)))/(n*k)-0.5*log(2*pi);

if(~isempty(validation))
    
    n = sum(validation);

    [PHI,lnBeta] = getPHI(X(validation,:),omega(validation,:),theta,model);
    
    nu = zeros(n,k);
    
    for i=1:k
        nu(:,i) = sum(PHI.*(PHI*SIGMAi(:,:,i)),2);
    end
    
    sigma = nu+exp(-lnBeta);

    delta = PHI*w-Y(validation,:);
    
    validRMSE = sqrt(sum(bsxfun(@times,delta.^2,omega(validation)))/(n*k));
    validLL = sum(sum(-0.5*delta.^2./sigma-0.5*log(sigma)))/(n*k)-0.5*log(2*pi);
    
end

end
