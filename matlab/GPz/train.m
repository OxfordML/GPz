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

learnPsi = model.learnPsi;

n = size(X,1);

pnames =    { 'maxIter' 'maxAttempts' 'omega'     'training'     'validation'   'Psi'};
defaults =  {  200      inf         ones(n,1)   true(n,1)      []   []};

[maxIter,maxAttempts,omega,training,validation,Psi]  = internal.stats.parseArgs(pnames, defaults, varargin{:});

Y = bsxfun(@minus,Y,muY);

X = bsxfun(@minus,X,muX);
X = bsxfun(@rdivide,X,sdX);


if(~learnPsi&&~isempty(Psi))
    Psi = fixSx(Psi,n,sdX,method);
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


theta = best_theta;
[~,~,w,iSigma_w] = f(theta);

model.best.theta = theta;
model.best.w = w;
model.best.iSigma_w = iSigma_w;


end