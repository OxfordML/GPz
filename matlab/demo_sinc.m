
rng(1)

addpath GPz/

addpath(genpath('minFunc_2012/'))    % path to minfunc

%%%%%%%%%%%%%% Model options %%%%%%%%%%%%%%%%

method = 'VL';               % select method, options = GL, VL, GC, GD, VD and VC
m = 200;                     % number of basis functions to use

%%%%%%%%%%%%%% Generate Data %%%%%%%%%%%%%%%%

n = 4000;
X  = linspace(-10,10,n)';
Xs = X;

X = X(X<-6|X>-3);

f_noise = 0.01+3*sin(X).*(1+exp(-0.1*X)).^-1;

Y = 10*sinc(2*X)+randn(size(X)).*f_noise.^2;

%%%%%%%%%%%%%% Start Script %%%%%%%%%%%%%%%%

% optain an initial model using the default options
model = init(X,Y,method,m);

% train the model using the default options
model = train(model,X,Y);

% use the model to generate predictions for the test set
[mu,sigma,variance,noise,PHI] = predict(Xs,model);

%%%%%%%%%%%%%% Display Results %%%%%%%%%%%%%%%% 

hold all;
f = [mu+2*sqrt(sigma); flip(mu-2*sqrt(sigma))];
h1 = fill([Xs; flip(Xs)], f, [0.85 0.85 0.85]);
plot(X,Y,'b.');


w = model.best.w;
SIGMAi = model.best.SIGMAi;
muY = model.muY;

[U,S] = svd(SIGMAi);

R = U*sqrt(S);

k = 20;
ws = bsxfun(@plus,R*randn(length(w),k),w);

mus = PHI*ws+muY;

plot(Xs,mus);
h4 = plot(Xs,mu,'r-','LineWidth',4);

axis([-10    10   -13    17]);
