rng(1)

addpath('GPz/');
addpath(genpath('minFunc_2012/'))


fx = @(x) sinc(x); % true function
sx = @(x) 0.05+(1./(1+exp(-0.2*x))).*(1+sin(2*x))*0.2; % true output noise function

%%%%%%% create dataset %%%%%%%
n = 10000;   % number of samples
X  = linspace(-10,10,n)'; % input
X = X(X<-7|X>-2); n = size(X,1); % create a gap

E = 0.5;  % desired mean of the input noise variance
V = 0.25; % desired variance of the input noise variance

% The parameters of a gamma distribution with the desired mean and variance
a = E^2/V; b = V/E;

Psi = gamrnd(a,b,size(X)); % sample from the gamma distribution with mean=E and variance=V

Xn = X+randn(size(X)).*sqrt(Psi);  % create a noisy input
Yn = fx(X)+randn(size(X)).*sx(X);  % create a noisy output
%%%%%%% model options and fitting %%%%%%%

m=100; % number of basis functions, the more the better but also the slower
method = 'VD';  % covariance types: G=Global, V=Variable, L=Length-scale, D=Diagonal and C=Covariance. For example, VD is Variable Diagonal.
maxIter = 500; % maximum number of iterations
maxAttempt = 200; % maximum number of attempts before stoping if no improvment is noticed in the validation set 
normalise = true; % subtract means and divid by the standard deviations.
heteroscedastic = true; % learn a heteroscedastic noise model
joint = true; % joint prior mean function optimisation

[training,validation,testing] = sample(n,0.6,0.2,0.2); % split data into training, validation and testing

%%%%%%% IMPORTANT NOTE %%%%%%%
% Psi is the input variance and its size depends on the dimensionality of
% the input and the chosen method. For GL, VL, GD and VD Psi is assumed to
% be (n x d), where n is the number of samples and d is the dimensionality of
% the input wheras for GC and VC, Psi is assumed to be (d x d x n). If the size
% of Psi does not match the covariance type, then the code will convert it so that
% they match. 

% Psi = []; % if you want to treat the points as exact, or simply remove
% the options 'Psi' from the input in the following three lines

model = init(Xn,Yn,method,m,'normalise',normalise,'heteroscedastic',heteroscedastic,'joint',joint,'training',training,'Psi',Psi);
model = train(model,Xn,Yn,'maxIter',maxIter,'maxAttempt',maxAttempt,'training',training,'validation',validation,'Psi',Psi);

% [mu,sigma,nu,beta_i,gamma] = predict(X(testing,:),model,'Psi',Psi(testing,:)); % generate predictions for the test set

% mu     = the best point estimate
% nu     = variance due to data density
% beta_i = variance due to output noise
% gamma  = variance due to input noise
% sigma  = nu+beta_i+gamma

% compute any metrics here, e.g. RMSE

%%%%%%% Display %%%%%%%

Xs = linspace(-15,15,1000)';

[mu,sigma,nu,beta_i,gamma,PHI,w,iSigma_w] = predict(Xs,model); % generate predictions, note that this will use the model with the best score on the validation set
% [mu,sigma,nu,beta_i,gamma,PHI,w,iSigma_w] = predict(Xs,model,'whichSet','last'); % this will use the model with the best score on the training set

hold on;

f = [mu+2*sqrt(sigma); flip(mu-2*sqrt(sigma))];
h1 = fill([Xs; flip(Xs)], f, [0.85 0.85 0.85]);
plot(Xn,Yn,'b.');

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
