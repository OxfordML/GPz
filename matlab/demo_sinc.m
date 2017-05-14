rng(1); % fix random seed
addpath GPz/ % path to GPz
addpath(genpath('minFunc_2012/'))       % path to minfunc
 
%%%%%%%%%%%%%% Model options %%%%%%%%%%%%%%%%
 
m = 100;                                % number of basis functions to use [required]
 
method = 'VD';                          % select a method, options = GL, VL, GD, VD, GC and VC [required]
 
joint = true;                           % jointly learn a prior linear mean-function [default=true]
 
heteroscedastic = true;                 % learn a heteroscedastic noise process, set to false if only interested in point estimates [default=true]
 
normalize = true;                       % pre-process the input by subtracting the means and dividing by the standard deviations [default=true]

maxIter = 500;                          % maximum number of iterations [default=200]
maxAttempts = 50;                       % maximum iterations to attempt if there is no progress on the validation set [default=infinity]
 
 
trainSplit = 0.2;                       % percentage of data to use for training
validSplit = 0.2;                       % percentage of data to use for validation
testSplit  = 0.6;                       % percentage of data to use for testing

inputNoise = true;                      % false = use mag errors as additional inputs, true = use mag errors as additional input noise

%%%%%%%%%%%%%% Create dataset %%%%%%%%%%%%%%

fx = @(x) sinc(x); % true function
sx = @(x) 0.05+(1./(1+exp(-0.2*x))).*(1+sin(2*x))*0.2; % true output noise function

n = 10000;   % number of samples
X  = linspace(-10,10,n)'; % input
X = X(X<-7|X>-2); n = size(X,1); % create a gap

Y = fx(X)+randn(size(X)).*sx(X);  % add noise to output

if(inputNoise)
    % add noise to input
    
    E = 0.5;  % desired mean of the input noise variance
    V = 0.25; % desired variance of the input noise variance

    % The parameters of a gamma distribution with the desired mean and variance
    a = E^2/V; b = V/E;

    Psi = gamrnd(a,b,size(X)); % sample from the gamma distribution with mean=E and variance=V
    
    X = X+randn(size(X)).*sqrt(Psi);  % create a noisy input
end

%%%%%%%%%%%%%% Fit the model %%%%%%%%%%%%%%

% split data into training, validation and testing
[training,validation,testing] = sample(n,trainSplit,validSplit,testSplit); 

if(inputNoise) 
    % initialize the model
    model = init(X,Y,method,m,'normalize',normalize,'heteroscedastic',heteroscedastic,'joint',joint,'training',training,'Psi',Psi);

    % train the model
    model = train(model,X,Y,'maxIter',maxIter,'maxAttempt',maxAttempts,'training',training,'validation',validation,'Psi',Psi);

    % use the model to generate predictions for the test set
    [mu,sigma,nu,beta_i,gamma] = predict(X(testing,:),model,'Psi',Psi(testing,:)); % generate predictions for the test set
else
    % initialize the model
    model = init(X,Y,method,m,'normalize',normalize,'heteroscedastic',heteroscedastic,'joint',joint,'training',training);

    % train the model
    model = train(model,X,Y,'maxIter',maxIter,'maxAttempt',maxAttempts,'training',training,'validation',validation);

    % use the model to generate predictions for the test set
    [mu,sigma,nu,beta_i,gamma] = predict(X(testing,:),model); % generate predictions for the test set
end

% mu     = the best point estimate
% nu     = variance due to data density
% beta_i = variance due to output noise
% gamma  = variance due to input noise
% sigma  = nu+beta_i+gamma

%%%%%%%%%%%%%% Display %%%%%%%%%%%%%%
Xs = linspace(-15,15,1000)';

[mu,sigma,nu,beta_i,gamma,PHI,w,iSigma_w] = predict(Xs,model,'Psi',[]); % generate predictions, note that this will use the model with the best score on the validation set
% [mu,sigma,nu,beta_i,gamma,PHI,w,iSigma_w] = predict(Xs,model,'whichSet','last'); % this will use the model with the best score on the training set

hold on;

f = [mu+2*sqrt(sigma); flip(mu-2*sqrt(sigma))];
h1 = fill([Xs; flip(Xs)], f, [0.85 0.85 0.85]);
plot(X,Y,'b.');

muY = model.muY;

[U,S] = svd(iSigma_w);

R = U*sqrt(S);

k = 20;
ws = bsxfun(@plus,R*randn(length(w),k),w);

mus = PHI*ws;
mus = bsxfun(@plus,mus,muY);

plot(Xs,mus);

h3 = plot(Xs,fx(Xs),'k-','LineWidth',2);
h2 = plot(Xs,mu,'r-','LineWidth',2);

axis tight;

legend([h1 h2 h3], {'95\%','$\mathbf{f}_{*}$','$\mbox{sinc}(x)$'},'FontSize',18,'Location','NorthWest','interpreter','latex');

xlabel('$x$','interpreter','latex','FontSize',30);
ylabel('$y$','interpreter','latex','FontSize',30);
