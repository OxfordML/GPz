function model = init(X,Y,method,m,varargin)

[n,d] = size(X);
k = size(Y,2);

pnames =    { 'heteroscedastic' 'normalize'   'omega'    'training'   'Psi' 'joint'};
defaults =  { true                true          ones(n,1)   true(n,1)   [] true};


[heteroscedastic,normalize,omega,training,Psi,joint]  = internal.stats.parseArgs(pnames, defaults, varargin{:});

model.k = k;
model.m = m;
model.method = method;
model.heteroscedastic = heteroscedastic;
model.joint = joint;

if(normalize)
    muX = mean(X(training,:));
    sdX = std(X(training,:));
else
    muX = zeros(1,d);
    sdX = ones(1,d);
end

muY = mean(Y(training,:));

model.d = d;
model.sdX = sdX;
model.muX = muX;
model.muY = muY;

Y = bsxfun(@minus,Y,muY);

X = bsxfun(@minus,X,muX);
X = bsxfun(@rdivide,X,sdX);

learnPsi = ischar(Psi);

if(~learnPsi&&~isempty(Psi))
    Psi = fixSx(Psi,n,sdX,method);
end

b = log(var(Y(training,:)));
lnAlpha = -log(var(Y(training,:)));

if(joint)
    lnAlpha = repmat(lnAlpha,m+d+1,1);
else
    lnAlpha = repmat(lnAlpha,m,1);
end

[mu,~,~,Vi] = pca(X(training,:),1);
P = (rand(m,d)-0.5)*sqrt(12);
P = bsxfun(@plus,P*Vi,mu);

lambda = sqrt(2*nthroot(m,d)./mean(Dxy(X(training,:),P)));

switch(method)
    case 'GL'
        Lambda = mean(lambda);
    case 'VL'
        Lambda = ones(1,m).*lambda;
    case 'GD'
        Lambda = ones(1,d)*mean(lambda);
    case 'VD'
        Lambda = zeros(m,d);
        for j=1:m
            Lambda(j,:) = ones(1,d)*lambda(j);
        end
    case 'GC'
        Lambda = eye(d)*mean(lambda);
    case 'VC'
        Lambda = zeros(d,d,m);
        for j=1:m
            Lambda(:,:,j) = eye(d)*lambda(j);
        end
end

theta = [P(:);Lambda(:);lnAlpha(:);b(:)];

f = @(params) GPz(params,model,X,Y,Psi,omega,training,[]);


if(heteroscedastic)
    
    v = zeros(m,k);
    lnTau = zeros(m,k);

    theta = [theta;v(:);lnTau(:)];
    
end

if(learnPsi)
    
    if(method(2)=='C')
        S = eye(d)/mean(lambda);
    else
        S = ones(1,d)/mean(lambda);
    end            
    
    theta = [theta;S(:)];
    
end

[~,~,w,iSigma_w] = f(theta);

last.theta = theta;
last.w = w;
last.iSigma_w = iSigma_w;


best = last;

best.LL = -inf;

model.last = last;
model.best = best;

model.learnPsi = learnPsi;


end

