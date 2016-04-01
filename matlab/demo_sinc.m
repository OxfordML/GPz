
rng(1)

addpath GPz/

addpath(genpath('minFunc_2012/'))    % path to minfunc

%%%%%%%%%%%%%%%%%%%%%%%%%%% Model options %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

method = 'VL';               % options include:
                             %   GL: global length-scale
                             %   VL: variable length-scales
                             %   GD: global diagonal covariance
                             %   VD: variable diagonal covariances
                             %   GC: global full covariance
                             %   VC: variable full covariances
                             
m = 200;                     % number of basis functions to use

%%%%%%%%%%%%%%%%%%%%%%%%%%% Generate Data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = 4000;
X  = linspace(-10,10,n)';   % training set
X = X(X<-6|X>-3);           % create a gap
Xs = linspace(-12,12,n)';   % test set

f_noise = 0.01+3*sin(X)./(1+exp(-0.1*X));     % true function

Y = 10*sinc(2*X)+randn(size(X)).*f_noise.^2;  % true noise function

%%%%%%%%%%%%%%%%%%%%%%%%%%% Start Script %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% optain an initial model using the default options
model = init(X,Y,method,m);

% train the model using the default options
model = train(model,X,Y);

% use the model to generate predictions for the test set
[mu,sigma,nu,beta_i,PHI] = predict(Xs,model);

%%%%%%%%%%%%%%%%%%%%%%%%%% Display Results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

hold all;

f = [mu+2*sqrt(sigma); flip(mu-2*sqrt(sigma))];
h1 = fill([Xs; flip(Xs)], f, [0.85 0.85 0.85]);
plot(X,Y,'b.');


w = model.best.w;
SIGMAi = model.best.SIGMAi;
muY = model.muY;
wL = model.wL;

[U,S] = svd(SIGMAi);

R = U*sqrt(S);

k = 20;
ws = bsxfun(@plus,R*randn(length(w),k),w);

mus = PHI*ws;
mus = bsxfun(@plus,mus,Xs*wL);
mus = bsxfun(@plus,mus,muY);

plot(Xs,mus);
h2 = plot(Xs,mu,'r-','LineWidth',4);

legend([h1 h2], {'$\pm 2\sigma_{*}$','$\mu_{*}$'},'FontSize',14,'Location','NorthWest','interpreter','latex');
