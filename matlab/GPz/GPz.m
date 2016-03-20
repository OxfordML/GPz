function [nlogML,grad,w,SIGMAi,PHI,lnBeta] = GPz(theta,method,m,X,Y,omega,joint,heteroscedastic,training,validation)

global trainRMSE
global trainLL

global validRMSE
global validLL

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
[GAMMA,dGAMMA,g_dim] = getGamma(theta,method,m,d);
lnAlpha = theta(m*d+g_dim+1:m*d+g_dim+a_dim);
b = theta(m*d+g_dim+a_dim+1);

PHI = zeros(n,m);
for j = 1:m
    Delta = bsxfun(@minus,X(training,:),P(j, :));
    PHI(:,j) = exp(-0.5*sum(power(Delta*GAMMA{j}',2),2));
end

lnBeta = b+log(omega(training,:));

if(heteroscedastic)
    u = theta(m*d+g_dim+a_dim+1+1:m*d+g_dim+a_dim+1+m);
    lnBeta = PHI*u+lnBeta;
end

if(joint)
    PHI = [PHI X(training,:) ones(n,1)];
end

if(isempty(Y))
    nlogML = 0;
    grad = 0;
    w = 0;
    SIGMAi = 0;
    return
end

beta = exp(lnBeta);

alpha = exp(lnAlpha);

PSi = bsxfun(@times,PHI,beta);

SIGMA = PSi'*PHI+diag(alpha);

L = chol(SIGMA);

SIGMAi = L\inv(L)';

logdet = 2*sum(log(diag(L)));

variance = sum(PHI.*(PHI*SIGMAi),2);

w = SIGMAi*PSi'*Y(training,:);

delta = PHI*w-Y(training,:);

nlogML = -0.5*sum(delta.^2.*beta)-0.5*sum(w.^2.*alpha)+0.5*sum(lnAlpha)+0.5*sum(lnBeta)-0.5*logdet-(n/2)*log(2*pi);


if(nargout>2)
    grad = 0;
    return
end

dbeta = -0.5*beta.*(delta.^2+variance)+0.5;
db = sum(dbeta);


if(heteroscedastic)
    lnEta = theta(m*d+g_dim+a_dim+1+m+1:m*d+g_dim+a_dim+1+m+m);
    eta = exp(lnEta);
    nlogML = nlogML-0.5*sum(u.^2.*eta)+0.5*sum(lnEta);
    du = PHI(:,1:m)'*dbeta-u.*eta;
    dlnEta = -0.5*eta.*u.^2+0.5;
    dlnPHI = (dbeta*u'-(delta.*beta)*w(1:m)'-PSi*SIGMAi(:,1:m)).*PHI(:,1:m);    
else
    dlnPHI = (-(delta.*beta)*w(1:m)'-PSi*SIGMAi(:,1:m)).*PHI(:,1:m);
end

dP = zeros(size(P));

for j=1:m
    Delta = bsxfun(@minus,X(training,:),P(j,:));
    dP(j,:) = dlnPHI(:,j)'*Delta*GAMMA{j}'*GAMMA{j};
    dGj = -GAMMA{j}*bsxfun(@times,Delta,dlnPHI(:,j))'*Delta;
    switch(method)
        case 'GL'
            dGAMMA = dGAMMA+sum(diag(dGj));
        case 'VL'
            dGAMMA(j) = sum(diag(dGj));
        case 'GD'
            dGAMMA = dGAMMA+diag(dGj);
        case 'VD'
            dGAMMA(:,j) = diag(dGj);
        case 'GC'
            dGAMMA = dGAMMA+dGj;
        case 'VC'
            dGAMMA(:,:,j) = dGj;
            
    end
end

dwda = -SIGMAi*(alpha.*w);

dlnAlpha = -(dwda*(beta.*delta)'.*PHI')*ones(n,1)-alpha.*w.*dwda-0.5*alpha.*w.^2-0.5*diag(SIGMAi).*alpha+0.5;

grad = [dP(:);dGAMMA(:);dlnAlpha(:);db];

if(heteroscedastic)
    grad = [grad;du(:);dlnEta(:)];
end


nlogML = -nlogML/n;
grad = -grad/n;

sigma = variance+exp(-lnBeta);

trainRMSE = sqrt(mean(omega(training).*delta.^2));
trainLL = mean(-0.5*delta.^2./sigma-0.5*log(sigma))-0.5*log(2*pi);

if(~isempty(validation))
    
    n = sum(validation);

    phi = zeros(n,m);
    for j = 1:m
        Delta = bsxfun(@minus,X(validation,:),P(j, :));
        phi(:,j) = exp(-0.5*sum(power(Delta*GAMMA{j}',2),2));
    end

    lnSigma = -b-log(omega(validation));
    
    if(heteroscedastic)
        lnSigma = lnSigma-phi*u;
    end

    if(joint)
        phi = [phi X(validation,:) ones(n,1)];
    end
    
    sigma = sum(phi.*(phi*SIGMAi),2)+exp(lnSigma);

    delta = phi*w-Y(validation,:);

    validRMSE = sqrt(mean(omega(validation).*delta.^2));
    validLL = mean(-0.5*delta.^2./sigma-0.5*log(sigma))-0.5*log(2*pi);
    
end

end
