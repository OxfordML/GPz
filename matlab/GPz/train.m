function model = train(model,X,Y,varargin)

clear global;

global best_valid
global best_theta

best_theta = model.best.theta;
best_valid = model.best.LL;

theta = model.last.theta;

method = model.method;
muY = model.muY;

muX = model.muX;
sdX = model.sdX;

[n,d] = size(X);

m = model.m;
k = model.k;
g_dim = model.g_dim;

pnames =    { 'maxIter' 'maxAttempts' 'omega'     'training'     'validation'   'Psi'};
defaults =  {  200      inf         ones(n,1)   true(n,1)      []   []};

[maxIter,maxAttempts,omega,training,validation,Psi]  = internal.stats.parseArgs(pnames, defaults, varargin{:});

Y = bsxfun(@minus,Y,muY);

X = bsxfun(@minus,X,muX);
X = bsxfun(@rdivide,X,sdX);


if(~isempty(Psi))
    Psi = fixPsi(Psi,n,sdX,method);
end

f = @(params) GPz(params,model,X,Y,Psi,omega,training,validation);

options.method = 'lbfgs';
options.display = 'off';
options.maxIter = maxIter;
options.MaxFunEvals = inf;
options.outputFcn = @(theta,iterationType,i,funEvals,f,t,gtd,g,dir,optCond,varargin) callBack(theta,iterationType,i,funEvals,f,t,gtd,g,dir,optCond,maxIter,maxAttempts,isempty(validation),varargin);

theta = minFunc(f,theta,options);

% options = optimoptions('fminunc','MaxIter',maxIter,'Algorithm','quasi-newton','GradObj','on','OutputFcn', @(theta,optimValues,state) outputFun(theta,optimValues,state,maxAttempts,isempty(validation)));
% theta = fminunc(f,theta,options);

[~,~,w,iSigma_w] = f(theta);


model.last.theta = theta;
model.last.w = w;
model.last.iSigma_w = iSigma_w;
model.last.priors = getPrior(X,Psi,theta,model,training);
model.last.P = reshape(theta(1:m*d),m,d);

if(model.heteroscedastic)
    v = reshape(theta(m*d+g_dim+m*k+k+1:m*d+g_dim+m*k+k+m*k),m,k);
    model.last.v = v;
end


theta = best_theta;
[~,~,w,iSigma_w] = f(theta);

model.best.theta = theta;
model.best.w = w;
model.best.iSigma_w = iSigma_w;
model.best.priors = getPrior(X,Psi,theta,model,training);
model.best.P = reshape(theta(1:m*d),m,d);

if(model.heteroscedastic)
    v = reshape(theta(m*d+g_dim+m*k+k+1:m*d+g_dim+m*k+k+m*k),m,k);
    model.best.v = v;
end

end