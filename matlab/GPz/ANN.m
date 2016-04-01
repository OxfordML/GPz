function [nlogML,grad,w,SIGMAi,PHI,lnBeta] = ANN(theta,model,X,Y,omega,training,validation)

layers = model.layers;

k = model.k;
m = layers(end);
n_layers = length(layers)-1;

joint = model.joint;
heteroscedastic = model.heteroscedastic;

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

PHI = cell(n_layers+1,1);

PHI{1} = X(training,:);

W = cell(n_layers,1);
bias = cell(n_layers,1);

ind = cumsum([0 layers(1:end-1).*layers(2:end)+layers(2:end)]);

for i=1:n_layers
    
    W{i} = reshape(theta(ind(i)+1:ind(i)+layers(i)*layers(i+1)),layers(i),layers(i+1));
    bias{i} = theta(ind(i)+layers(i)*layers(i+1)+1:ind(i)+layers(i)*layers(i+1)+layers(i+1))';
    
    PHI{i+1} = tanh(bsxfun(@plus,PHI{i}*W{i},bias{i}));
end


lnAlpha = reshape(theta(ind(end)+1:ind(end)+a_dim*k),a_dim,k);
b = theta(ind(end)+a_dim*k+1:ind(end)+a_dim*k+k)';

lnBeta = bsxfun(@plus,log(omega(training,:)),repmat(b,n,1));

if(heteroscedastic)
    u = reshape(theta(ind(end)+a_dim*k+k+1:ind(end)+a_dim*k+k+m*k),m,k);
    lnBeta = PHI{end}*u+lnBeta;
end

if(joint)
    PHI{end} = [PHI{end} X(training,:) ones(n,1)];
end

if(isempty(Y))
    nlogML = 0;
    grad = 0;
    w = 0;
    SIGMAi = 0;
    PHI = PHI{end};
    return
end

beta = exp(lnBeta);

alpha = exp(lnAlpha);

nu = zeros(n,k);
w = zeros(a_dim,k);
dwda = zeros(a_dim,k);
SIGMAi = zeros(a_dim,a_dim,k);
logdet = zeros(1,k);
dPHI = zeros(n,m);
dlnAlpha = zeros(a_dim,k);

for i=1:k
    BxPHI = bsxfun(@times,PHI{end},beta(:,i));

    SIGMA = BxPHI'*PHI{end}+diag(alpha(:,i));

    [U,S,~] = svd(SIGMA);

    SIGMAi(:,:,i) = (U/S)*U';
    logdet(i) = sum(log(diag(S)));

    nu(:,i) = sum(PHI{end}.*(PHI{end}*SIGMAi(:,:,i)),2);

    w(:,i) = SIGMAi(:,:,i)*BxPHI'*Y(training,i);

    dwda(:,i) = -SIGMAi(:,:,i)*(alpha(:,i).*w(:,i));
    
    dPHI = dPHI-(BxPHI*SIGMAi(:,1:m,i)).*(1-PHI{end}(:,1:m).^2);
    dlnAlpha(:,i) = -0.5*diag(SIGMAi(:,:,i)).*alpha(:,i);

end

delta = PHI{end}*w-Y(training,:);

dPHI = dPHI-((delta.*beta)*w(1:m,:)').*(1-PHI{end}(:,1:m).^2);

nlogML = -0.5*sum(delta.^2.*beta)-0.5*sum(w.^2.*alpha)+0.5*sum(lnAlpha)+0.5*sum(lnBeta)-0.5*logdet;

if(nargout>2)
    PHI = PHI{end};
    grad = 0;
    return
end

dbeta = -0.5*beta.*(delta.^2+nu)+0.5;
db = sum(dbeta);

if(heteroscedastic)
    lnEta = reshape(theta(ind(end)+a_dim*k+k+m*k+1:ind(end)+a_dim*k+k+m*k+m*k),m,k);
    eta = exp(lnEta);
    nlogML = nlogML-0.5*sum(u.^2.*eta)+0.5*sum(lnEta);
    du = PHI{end}(:,1:m)'*dbeta-u.*eta;
    dlnEta = -0.5*eta.*u.^2+0.5;
    dPHI = dPHI+(dbeta*u').*(1-PHI{end}(:,1:m).^2);
end

dlnAlpha = dlnAlpha-(PHI{end}'*(beta.*delta)).*dwda-alpha.*w.*dwda-0.5*alpha.*w.^2+0.5;

grad = zeros(ind(end),1);

for i=n_layers:-1:1
   
    dW = PHI{i}'*dPHI;
    dbias = sum(dPHI);
    
    grad(ind(i)+1:ind(i)+layers(i)*layers(i+1)) = dW(:);
    grad(ind(i)+layers(i)*layers(i+1)+1:ind(i)+layers(i)*layers(i+1)+layers(i+1)) = dbias(:);

    dPHI = (dPHI*W{i}').*(1-PHI{i}.^2);
end

grad = [grad;dlnAlpha(:);db(:)];

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

    PHI = X(validation,:);
    
    for i=1:n_layers
        PHI = tanh(bsxfun(@plus,PHI*W{i},bias{i}));
    end
    
    lnBeta = bsxfun(@plus,log(omega(validation,:)),repmat(b,n,1));
    
    if(heteroscedastic)
        lnBeta = PHI*u+lnBeta;
    end

    if(joint)
        PHI = [PHI X(validation,:) ones(n,1)];
    end
    
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
