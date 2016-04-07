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

[PHI,lnBeta,GAMMA,lnPHI] = getPHI(X(training,:),theta,model);

lnBeta = bsxfun(@plus,lnBeta,log(omega(training,:)));

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

    [U,S] = svd(SIGMA);
    
    SIGMAi(:,:,i) = (U/S)*U';
    logdet(i) = sum(log(abs(diag(S))));
    
    nu(:,i) = sum(PHI.*(PHI*SIGMAi(:,:,i)),2);

    w(:,i) = SIGMAi(:,:,i)*BxPHI'*Y(training,i);

    dwda(:,i) = -SIGMAi(:,:,i)*(alpha(:,i).*w(:,i));
    
    dlnPHI = dlnPHI-(BxPHI*SIGMAi(:,1:m,i)).*PHI(:,1:m);
    dlnAlpha(:,i) = -0.5*diag(SIGMAi(:,:,i)).*alpha(:,i);

end

delta = PHI*w-Y(training,:);

beta_x_delta = beta.*delta;

nlogML = -0.5*sum(beta_x_delta.*delta)+0.5*sum(lnBeta)-0.5*sum(alpha.*w.^2)+0.5*sum(lnAlpha)-0.5*logdet-0.5*n*log(2*pi);

if(nargout>2)
    grad = 0;
    return
end

dlnAlpha = dlnAlpha-(PHI'*beta_x_delta).*dwda-alpha.*w.*dwda-0.5*alpha.*w.^2+0.5;
dlnPHI= dlnPHI-(beta_x_delta*w(1:m,:)').*PHI(:,1:m);

dbeta = -0.5*(beta_x_delta.*delta+beta.*nu)+0.5;
db = sum(dbeta);

if(heteroscedastic)
    u = reshape(theta(m*d+g_dim+a_dim*k+k+1:m*d+g_dim+a_dim*k+k+m*k),m,k);
    lnEta = reshape(theta(m*d+g_dim+a_dim*k+k+m*k+1:m*d+g_dim+a_dim*k+k+m*k+m*k),m,k);
    eta = exp(lnEta);
    nlogML = nlogML-0.5*sum(u.^2.*eta)+0.5*sum(lnEta)-0.5*m*log(2*pi);
    du = PHI(:,1:m)'*dbeta-u.*eta;
    dlnEta = -0.5*eta.*u.^2+0.5;
    dlnPHI = dlnPHI+(dbeta*u').*PHI(:,1:m);    
end

dP = zeros(size(P));
dGAMMA = zeros(size(GAMMA));

for j=1:m
    
    Delta = bsxfun(@minus,X(training,:),P(j,:));
    dlnPHIjTxDelta = dlnPHI(:,j)'*Delta;
    
    switch(method)
        case 'GL'
            dP(j,:) = dlnPHIjTxDelta*GAMMA^2;
            dGAMMA = dGAMMA+2*sum(dlnPHI(:,j).*lnPHI(:,j))*GAMMA^-1;
        case 'VL'
            dP(j,:) = dlnPHIjTxDelta*GAMMA(j)^2;
            dGAMMA(j) = 2*sum(dlnPHI(:,j).*lnPHI(:,j))*GAMMA(j)^-1;
        case 'GD'
            dP(j,:) = dlnPHIjTxDelta.*GAMMA.^2;
            dGAMMA = dGAMMA-sum((dlnPHI(:,j)*GAMMA).*Delta.^2);
        case 'VD'
            dP(j,:) = dlnPHIjTxDelta.*GAMMA(j,:).^2;
            dGAMMA(j,:) = -sum((dlnPHI(:,j)*GAMMA(j,:)).*Delta.^2);
        case 'GC'
            dP(j,:) = dlnPHIjTxDelta*(GAMMA'*GAMMA);
            dGAMMA = dGAMMA-GAMMA*bsxfun(@times,Delta,dlnPHI(:,j))'*Delta;
        case 'VC'
            dP(j,:) = dlnPHIjTxDelta*(GAMMA(:,:,j)'*GAMMA(:,:,j));
            dGAMMA(:,:,j) = -GAMMA(:,:,j)*bsxfun(@times,Delta,dlnPHI(:,j))'*Delta;
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

    [PHI,lnBeta] = getPHI(X(validation,:),theta,model);
    
    lnBeta = bsxfun(@plus,lnBeta,log(omega(validation,:)));
    
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
