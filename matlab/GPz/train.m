function model = train(model,X,Y,varargin)

clear global;

global best_valid
global best_theta

best_theta = model.best.theta;
best_valid = model.best.LL;

theta = model.last.theta;
muY = model.muY;
muX = model.muX;
T = model.T;

n = size(X,1);

pnames =    { 'maxIter' 'maxAttempts' 'omega'     'training'     'validation'};
defaults =  {  200      inf         ones(n,1)   true(n,1)      []};

[maxIter,maxAttempts,omega,training,validation]  = internal.stats.parseArgs(pnames, defaults, varargin{:});

X = bsxfun(@minus,X,muX)*T;

Y = bsxfun(@minus,Y,muY);

if(strcmp(model.method,'ANN'))
    f = @(params) ANN(params,model,X,Y,omega,training,validation);
else
    f = @(params) GPz(params,model,X,Y,omega,training,validation);
end

options.method = 'lbfgs';
options.display = 'off';
options.maxIter = maxIter;
options.MaxFunEvals = inf;
options.outputFcn = @(theta,iterationType,i,funEvals,f,t,gtd,g,d,optCond) callBack(theta,iterationType,i,funEvals,f,t,gtd,g,d,optCond,maxIter,maxAttempts,isempty(validation));
start = tic;
theta = minFunc(f,theta,options);
toc(start)
[~,~,w,SIGMAi] = f(theta);


model.last.theta = theta;
model.last.w = w;
model.last.SIGMAi = SIGMAi;

[~,~,w,SIGMAi] = f(best_theta);

model.best.theta = best_theta;
model.best.w = w;
model.best.SIGMAi = SIGMAi;
model.best.LL = best_valid;

end

