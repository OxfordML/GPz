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

pnames =    { 'maxIter' 'maxAttempts' 'omega'     'training'     'validation'   'Psi' 'testing'};
defaults =  {  200      inf         ones(n,1)   true(n,1)      []   []    []};

[maxIter,maxAttempts,omega,training,validation,Psi,testing]  = internal.stats.parseArgs(pnames, defaults, varargin{:});

if(~any(validation(:)))
    validation = [];
end
if(isempty(omega))
    omega = ones(n,1);
end

Y = bsxfun(@minus,Y,muY);

X = bsxfun(@minus,X,muX);
X = bsxfun(@rdivide,X,sdX);


if(~isempty(Psi))
    Psi = fixPsi(Psi,n,sdX,method);
end

f = @(params) GPz(params,model,X,Y,Psi,omega,training,validation);

% Test scores are evaluated only by the progress callback, not the optimizer.
testFun = [];
if(any(testing(:)))
    testFun = @(theta) testMetrics(theta,f,model,X,Y,Psi,omega,testing);
end

options = optimoptions('fminunc','Algorithm','quasi-newton', ...
    'HessianApproximation',{'lbfgs',100},'SpecifyObjectiveGradient',true, ...
    'Display','off','MaxIterations',maxIter,'MaxFunctionEvaluations',inf, ...
    'OptimalityTolerance',1e-5,'StepTolerance',1e-9,'FunctionTolerance',1e-9, ...
    'OutputFcn',@(theta,optimValues,state) outputFun(theta,optimValues,state,maxAttempts,isempty(validation),f,maxIter,testFun));
theta = fminunc(f,theta,options);

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
model.best.LL = best_valid;
model.best.w = w;
model.best.iSigma_w = iSigma_w;
model.best.priors = getPrior(X,Psi,theta,model,training);
model.best.P = reshape(theta(1:m*d),m,d);

if(model.heteroscedastic)
    v = reshape(theta(m*d+g_dim+m*k+k+1:m*d+g_dim+m*k+k+m*k),m,k);
    model.best.v = v;
end

end

function [testRMSE,testLL] = testMetrics(theta,f,model,X,Y,Psi,omega,testing)
    
    [~,~,w] = f(theta);
    [PHI,~,lnBeta_i] = getPHI(X,Psi,theta,model,testing);
    
    beta = exp(-lnBeta_i);
    delta = PHI*w-Y(testing,:);
    
    n = sum(testing);
    k = model.k;
    
    testRMSE = sqrt(sum(sum(bsxfun(@times,delta.^2,omega(testing))))/(n*k));
    testLL = sum(sum(bsxfun(@times,-0.5*beta.*delta.^2+0.5*log(beta),omega(testing,:))))/(n*k)-0.5*log(2*pi);
end
