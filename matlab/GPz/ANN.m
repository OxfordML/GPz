function [nlogML,grad,w,SIGMAi,PHI,lnBeta] = ANN(theta,model,X,Y,omega,training,validation)

layers = model.layers;
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


lnAlpha = theta(ind(end)+1:ind(end)+a_dim);
b = theta(ind(end)+a_dim+1);

lnBeta = b+log(omega(training,:));

if(heteroscedastic)
    u = theta(ind(end)+a_dim+1+1:ind(end)+a_dim+1+m);
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

BxPHI = bsxfun(@times,PHI{end},beta);

SIGMA = BxPHI'*PHI{end}+diag(alpha);

[U,S,~] = svd(SIGMA);

SIGMAi = (U/S)*U';
logdet = sum(log(diag(S)));

variance = sum(PHI{end}.*(PHI{end}*SIGMAi),2);

w = SIGMAi*BxPHI'*Y(training,:);

delta = PHI{end}*w-Y(training,:);

nlogML = -0.5*sum(delta.^2.*beta)-0.5*sum(w.^2.*alpha)+0.5*sum(lnAlpha)+0.5*sum(lnBeta)-0.5*logdet-(n/2)*log(2*pi);

if(nargout>2)
    PHI = PHI{end};
    grad = 0;
    return
end

dbeta = -0.5*beta.*(delta.^2+variance)+0.5;
db = sum(dbeta);

if(heteroscedastic)
    lnEta = theta(ind(end)+a_dim+1+m+1:ind(end)+a_dim+1+m+m);
    eta = exp(lnEta);
    nlogML = nlogML-0.5*sum(u.^2.*eta)+0.5*sum(lnEta);
    du = PHI{end}(:,1:m)'*dbeta-u.*eta;
    dlnEta = -0.5*eta.*u.^2+0.5;
    dPHI = (dbeta*u'-(delta.*beta)*w(1:m)'-BxPHI*SIGMAi(:,1:m)).*(1-PHI{end}(:,1:m).^2);
else
    dPHI = (-(delta.*beta)*w(1:m)'-BxPHI*SIGMAi(:,1:m)).*(1-PHI{end}(:,1:m).^2);
end

dwda = -SIGMAi*(alpha.*w);

dlnAlpha = -(dwda*(beta.*delta)'.*PHI{end}')*ones(n,1)-alpha.*w.*dwda-0.5*alpha.*w.^2-0.5*diag(SIGMAi).*alpha+0.5;

grad = zeros(ind(end),1);

for i=n_layers:-1:1
   
    dW = PHI{i}'*dPHI;
    dbias = sum(dPHI);
    
    grad(ind(i)+1:ind(i)+layers(i)*layers(i+1)) = dW(:);
    grad(ind(i)+layers(i)*layers(i+1)+1:ind(i)+layers(i)*layers(i+1)+layers(i+1)) = dbias(:);

    dPHI = (dPHI*W{i}').*(1-PHI{i}.^2);
end

grad = [grad;dlnAlpha(:);db];

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

    PHI = X(validation,:);
    
    for i=1:n_layers
        PHI = tanh(bsxfun(@plus,PHI*W{i},bias{i}));
    end
    
    lnSigma = -b-log(omega(validation));
    
    if(heteroscedastic)
        lnSigma = lnSigma-PHI*u;
    end

    if(joint)
        PHI = [PHI X(validation,:) ones(n,1)];
    end
    
    sigma = sum(PHI.*(PHI*SIGMAi),2)+exp(lnSigma);

    delta = PHI*w-Y(validation,:);

    validRMSE = sqrt(mean(omega(validation).*delta.^2));
    validLL = mean(-0.5*delta.^2./sigma-0.5*log(sigma))-0.5*log(2*pi);
    
end

end
