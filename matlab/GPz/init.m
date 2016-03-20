function model = init(X,Y,method,m,varargin)

[n,d] = size(X);

pnames =    { 'heteroscedastic' 'joint' 'decorrelate'   'omega'    'training'};
defaults =  { true               true    false          ones(n,1)   true(n,1)};


[heteroscedastic,joint,decorrelate,omega,training]  = internal.stats.parseArgs(pnames, defaults, varargin{:});

muY = mean(Y(training,:));

Y = Y-muY;

if(decorrelate)
    [muX,T,Ti] = pca(X(training,:),1);
else
    muX = 0;
    T = 1;
    Ti = 1;
end

X = bsxfun(@minus,X,muX)*T;

    
[mu,~,Vi] = pca(X(training,:),1);

P = (rand(m,d)-0.5)*sqrt(12);
P = bsxfun(@plus,P*Vi,mu);


D = Dxy(X(training,:),P);

gamma = sqrt(2*nthroot(m,d)./mean(D));

varY = var(Y(training,:));

b = -log(varY);
lnAlpha = log(varY);

if(joint)
    lnAlpha = lnAlpha*ones(m+d+1,1);
else
    lnAlpha = lnAlpha*ones(m,1);
end

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

f = @(params) GPz(params,method,m,X,Y,omega,joint,heteroscedastic,training,[]);

theta = [P(:);GAMMA(:);lnAlpha(:);b];

if(heteroscedastic)
    u = zeros(m,1);
    lnEta = zeros(m,1);
    
    [~,~,w,~,PHI] = f([theta;u(:);lnEta(:)]);
    
    target = -log((Y(training)-PHI*w).^2)-b;
    lnEta = log(var(target))*ones(m,1);
    
%     u = (PHI(:,1:m)'*PHI(:,1:m)+diag(exp(lnEta)))\PHI(:,1:m)'*target;
    
    theta = [theta;u(:);lnEta(:)];
    
end

w = zeros(size(lnAlpha));
SIGMAi = diag(exp(-lnAlpha));

last.theta = theta;
last.w = w;
last.SIGMAi = SIGMAi;

best.theta = theta;
best.w = w;
best.SIGMAi = SIGMAi;
best.LL = -inf;

model.last = last;
model.best = best;

model.method = method;
model.joint = joint;
model.heteroscedastic = heteroscedastic;

model.d = d;
model.m = m;
model.muX = muX;
model.T = T;
model.Ti = Ti;
model.muY = muY;

end

