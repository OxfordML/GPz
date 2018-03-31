function model = init(X,Y,method,m,varargin)

[n,d] = size(X);
k = size(Y,2);

pnames =    { 'heteroscedastic' 'normalize'   'omega'    'training'   'Psi'};
defaults =  { true                true          ones(n,1)   true(n,1)   []};


[heteroscedastic,normalize,omega,training,Psi]  = internal.stats.parseArgs(pnames, defaults, varargin{:});

if(d==1)
    method(2) = 'L';
end

model.d = d;
model.k = k;
model.m = m;
model.method = method;
model.heteroscedastic = heteroscedastic;

if(normalize)
    Xz = X;
    missing = isnan(X);
    Xz(missing) = 0;

    counts = sum(~missing);

    muX = sum(Xz)./counts;

    sdX = sum(Xz.^2)./counts;
    sdX = sqrt(sdX-muX.^2);
else
    muX = zeros(1,d);
    sdX = ones(1,d);
end

muY = mean(Y(training,:));


model.sdX = sdX;
model.muX = muX;
model.muY = muY;

Y = bsxfun(@minus,Y,muY);

X = bsxfun(@minus,X,muX);
X = bsxfun(@rdivide,X,sdX);

if(~isempty(Psi))
    Psi = fixPsi(Psi,n,sdX,method);
end

b = log(var(Y(training,:)));
lnAlpha = repmat(-log(var(Y(training,:))),m,1);

[mu,sigmas,~,Vi] = pca(X(training,:),1);
P = (rand(m,d)-0.5)*sqrt(12);
P = bsxfun(@plus,P*Vi,mu);

Xl = fillLinear(X(training,:),mu,sigmas);
gamma = sqrt(0.5*nthroot(m,d)./mean(Dxy(Xl,P)));


switch(method)
    case 'GL'
        Gamma = mean(gamma);
    case 'VL'
        Gamma = ones(1,m).*gamma;
    case 'GD'
        Gamma = ones(1,d)*mean(gamma);
    case 'VD'
        Gamma = zeros(m,d);
        for j=1:m
            Gamma(j,:) = ones(1,d)*gamma(j);
        end
    case 'GC'
        Gamma = eye(d)*mean(gamma);
    case 'VC'
        Gamma = zeros(d,d,m);
        for j=1:m
            Gamma(:,:,j) = eye(d)*gamma(j);
        end
end

model.g_dim = length(Gamma(:));
theta = [P(:);Gamma(:);lnAlpha(:);b(:)];

f = @(params) GPz(params,model,X,Y,Psi,omega,training,[]);


if(heteroscedastic)
    
    v = zeros(m,k);
    lnTau = zeros(m,k);

    theta = [theta;v(:);lnTau(:)];
    
    last.v = v;
    
end

priors = ones(1,m)/m;
[~,~,w,iSigma_w] = f(theta);

last.theta = theta;
last.w = w;
last.iSigma_w = iSigma_w;
last.priors = priors;
last.P = P;




best = last;

best.LL = -inf;

model.last = last;
model.best = best;


end

