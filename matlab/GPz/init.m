function model = init(X,Y,method,m,varargin)

[n,d] = size(X);
k = size(Y,2);

pnames =    { 'heteroscedastic' 'joint' 'decorrelate'   'omega'    'training'};
defaults =  { true               true    false          ones(n,1)   true(n,1)};


[heteroscedastic,joint,decorrelate,omega,training]  = internal.stats.parseArgs(pnames, defaults, varargin{:});

model.k = k;
model.m = m(end);
model.method = method;
model.joint = joint;
model.heteroscedastic = heteroscedastic;

muY = mean(Y(training,:));

Y = bsxfun(@minus,Y,muY);

if(decorrelate||strcmp(method,'ANN'))
    [muX,T,Ti] = pca(X(training,:),1);
else
    muX = 0;
    T = 1;
    Ti = 1;
end

X = bsxfun(@minus,X,muX)*T;

lnA = -log(var(Y(training,:)));
lnB = -log(var(Y(training,:)));

theta = [lnA(:);lnB(:)];

options.method = 'lbfgs';
options.display = 'off';
options.maxIter = 50;

f = @(params) bayesianLinearRegression(params,X(training,:),Y(training,:),X(training,:)'*X(training,:),X(training,:)'*Y(training,:));

theta = minFunc(f,theta,options);

[~,~,wL] = f(theta);

model.wL = wL;

Y = bsxfun(@minus,Y,X*wL);

b = -log(var(Y(training,:)));
lnAlpha = -log(var(Y(training,:)));


if(joint)
    lnAlpha = repmat(lnAlpha,m(end)+d+1,1);
else
    lnAlpha = repmat(lnAlpha,m(end),1);
end

if(strcmp(method,'ANN'))
    
    layers = [d m];
    model.layers = layers;
    
    ind = cumsum([0 layers(1:end-1).*layers(2:end)+layers(2:end)]);
    
    theta = zeros(ind(end),1);

    PHI = X(training,:);
    
    for i=1:length(m)
        
        W = (2*rand(layers(i),layers(i+1))-1)/sqrt(layers(i));
        
        theta(ind(i)+1:ind(i)+layers(i)*layers(i+1)) = W(:);
        
        PHI = tanh(PHI*W);
    end
    
    theta = [theta;lnAlpha(:);b(:)];
    
    f = @(params) ANN(params,model,X,Y,omega,training,[]);
else
    [mu,~,Vi] = pca(X(training,:),1);
    P = (rand(m,d)-0.5)*sqrt(12);
    P = bsxfun(@plus,P*Vi,mu);

    D = Dxy(X(training,:),P);

    gamma = sqrt(2*nthroot(m,d)./mean(D));
    
    switch(method)
        case 'GL'
            GAMMA = mean(gamma);
        case 'VL'
            GAMMA = ones(1,m).*gamma;
        case 'GD'
            GAMMA = ones(d,1)*mean(gamma);
        case 'VD'
            GAMMA = bsxfun(@times,ones(d,m),gamma);
        case 'GC'
            GAMMA = eye(d)*mean(gamma);
        case 'VC'
            GAMMA = zeros(d,d,m);
            for j=1:m
                GAMMA(:,:,j) = eye(d)*gamma(j);
            end
    end

    theta = [P(:);GAMMA(:);lnAlpha(:);b(:)];
    
    f = @(params) GPz(params,model,X,Y,omega,training,[]);

end


if(heteroscedastic)
    u = zeros(m(end),k);
    lnEta = zeros(m(end),k);
    
    [~,~,w,~,PHI] = f([theta;u(:);lnEta(:)]);
    
    target = -bsxfun(@plus,log((Y(training,:)-PHI*w).^2),b);
    lnEta = repmat(log(var(target)),m(end),1);
    
%     u = (PHI(:,1:m)'*PHI(:,1:m)+diag(exp(lnEta)))\PHI(:,1:m)'*target;
    
    theta = [theta;u(:);lnEta(:)];
    
end

w = zeros(size(lnAlpha));

da = size(lnAlpha,1);
SIGMAi = zeros(da,da,k);
for i=1:k
    SIGMAi(:,:,i) = diag(exp(-lnAlpha(:,i)));
end

[~,~,w,SIGMAi] = f(theta);

last.theta = theta;
last.w = w;
last.SIGMAi = SIGMAi;

best.theta = theta;
best.w = w;
best.SIGMAi = SIGMAi;
best.LL = -inf;

model.last = last;
model.best = best;

model.d = d;
model.muX = muX;
model.T = T;
model.Ti = Ti;
model.muY = muY;

end

