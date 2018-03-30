rng(1); % fix random seed
addpath GPz/ % path to GPz
addpath(genpath('minFunc_2012/'))       % path to minfunc
 
%%%%%%%%%%%%%% Model options %%%%%%%%%%%%%%%%
 
m = 10;                                % number of basis functions to use [required]
 
method = 'VC';                          % select a method, options = GL, VL, GD, VD, GC and VC [required]
 
heteroscedastic = true;                 % learn a heteroscedastic noise process, set to false if only interested in point estimates [default=true]
 
normalize = true;                       % pre-process the input by subtracting the means and dividing by the standard deviations [default=true]

maxIter = 500;                          % maximum number of iterations [default=200]
maxAttempts = 50;                       % maximum iterations to attempt if there is no progress on the validation set [default=infinity]
 
 
trainSplit = 0.7;                       % percentage of data to use for training
validSplit = 0.3;                       % percentage of data to use for validation
testSplit  = 0;                       % percentage of data to use for testing

inputNoise = false;                     % false = use mag errors as additional inputs, true = use mag errors as additional input noise

percentage = 0.5;                       % percentage of data with a missing variable
%%%%%%%%%%%%%% Create dataset %%%%%%%%%%%%%%

mean1 = [10 0];
Sigma1 = [10 0;0 1];

mean2 = [10 10];
Sigma2 = [5 -3;-3 3];

mean3 = [5 5];
Sigma3 = [2 0;0 2];

X = [mvnrnd(mean1,Sigma1,1000);mvnrnd(mean2,Sigma2,1000);mvnrnd(mean3,Sigma3,1000)];

[n,d] = size(X);

PHI = [mvnpdf(X,mean1,Sigma1) mvnpdf(X,mean2,Sigma2) mvnpdf(X,mean3,Sigma3)];

w = [-9;6;3];
Y = PHI*w+randn(n,1)*0.01;

if(inputNoise)
    E = 0.1; % the mean of the input noise variance
    V = 1; % the variance of the input noise variance
    
    a = E^2/V;
    b = E/V;
    Psi = gamrnd(a,1/b,size(X));
    
    Xn = X+randn(size(X)).*sqrt(Psi); % this is the noisy version of the input
    
    % concert to covariances when using GC or VC
    if(method(2)=='C') 
        Psi = reshape([Psi(:,1) zeros(n,2) Psi(:,2)]',2,2,n);
    end
else
    
    Psi = [];
    Xn = X;
end


[n,d] = size(Xn);

% removing random variables from training
if(percentage>0)
    r = randperm(n)';
    psize = ceil(percentage*n/2); 

    Xn(r(1:psize),1) = nan; % remove the first variable from half the selected sample
    Xn(r(psize+1:2*psize),2) = nan; % remove the second variable from the other half the selected sample
end

%%%%%%%%%%%%%% Fit the model %%%%%%%%%%%%%%

% split data into training, validation and testing
[training,validation,testing] = sample(n,trainSplit,validSplit,testSplit);

% initialize the model
model = init(Xn,Y,method,m,'heteroscedastic',heteroscedastic,'normalize',normalize,'training',training,'Psi',Psi);

% train the model
model = train(model,Xn,Y,'maxIter',maxIter,'maxAttempt',maxAttempt,'training',training,'validation',validation,'Psi',Psi);


%%%%%%%%%%%%%% Visualize Results %%%%%%%%%%%%%%

% create 2D test data
[x,y] = meshgrid(linspace(min(X(:,1))-1,max(X(:,1))+1,100),linspace(min(X(:,2))-1,max(X(:,2))+1,100));
Xs = [x(:) y(:)];

[mu,sigma] = predict(Xs,model);

% Visualize prediction
figure;
subplot(2,3,1)
surf(x,y,reshape(mu,size(x)));colormap jet;axis tight;
hold on;
Xd = Xn;
Xd(isnan(Xd(:,1)),1) = min(Xs(:,1));
Xd(isnan(Xd(:,2)),2) = min(Xs(:,2));

plot3(Xd(training,1),Xd(training,2),Y(training),'.');
xlabel('$x$','interpreter','latex','FontSize',12);
ylabel('$y$','interpreter','latex','FontSize',12);
zlabel('$z$','interpreter','latex','FontSize',12);
title('Predicted Model','interpreter','latex','FontSize',12);

% Visualize ground truth
subplot(2,3,4)
PHI = [mvnpdf(Xs,mean1,Sigma1) mvnpdf(Xs,mean2,Sigma2) mvnpdf(Xs,mean3,Sigma3)];
truth = PHI*w;

surf(x,y,reshape(truth,size(x)));colormap jet;axis tight;
position = get(gcf,'Position');
position(1) = 0;
position(3) = 1200;
position(4) = 650;
set(gcf,'Position',position);
xlabel('$x$','interpreter','latex','FontSize',12);
ylabel('$y$','interpreter','latex','FontSize',12);
zlabel('$z$','interpreter','latex','FontSize',12);
title('Reference Model','interpreter','latex','FontSize',12);

%%%%%%%%%%%%%% Predict with missing variables %%%%%%%%%%%%%%

labels = ['x','y'];

for o=1:2
    
    % create a test set with only variable 'o' observed
    
    range_o = max(X(:,o))-min(X(:,o));
    Xo = linspace(min(X(:,o))-range_o/10,max(X(:,o))+range_o/10,1000)';
    
    % set missing variables to NaNs and observed variables to Xo
    Xs = nan(size(Xo,1),2); 
    Xs(:,o) = Xo;
    
    [mu,sigma] = predict(Xs,model);

    % plot the results
    subplot(2,3,o+1);
    hold on;
    f = [mu+2*sqrt(sigma); flip(mu-2*sqrt(sigma))];
    fill([Xo; flip(Xo)], f, [0.85 0.85 0.85]);
    plot(Xn(training,o),Y(training),'b.');
    plot(Xo,mu,'r-','LineWidth',2);
    
    xlabel(['$',labels(o),'$'],'interpreter','latex','FontSize',12);
    ylabel('$z$','interpreter','latex','FontSize',12);
    title('Predicted Model','interpreter','latex','FontSize',12);
    axis tight
    ax1 = gca;

    % build a reference model trained only on the observed variable to compare
    
    if(method(2)=='C')
        Psi_oo = squeeze(Psi(o,o,:));
    else
        Psi_oo = Psi(:,o);
    end
    
    removed = isnan(Xn(:,o));
    
    % build and train the reference model only on the observed variable
    ref_model = init(Xn(:,o),Y,method,m,'heteroscedastic',heteroscedastic,'normalize',normalize,'training',training&~removed,'Psi',Psi_oo);
    ref_model = train(ref_model,Xn(:,o),Y,'maxIter',maxIter,'maxAttempt',maxAttempt,'training',training&~removed,'validation',validation&~removed,'Psi',Psi_oo);

    % generate predictions using the reference model
    [mu,sigma] = predict(Xo,ref_model);

    % visualize the results
    subplot(2,3,o+4);
    hold on;
    f = [mu+2*sqrt(sigma); flip(mu-2*sqrt(sigma))];
    fill([Xo; flip(Xo)], f, [0.85 0.85 0.85]);
    plot(Xn(training,o),Y(training),'b.');
    plot(Xo,mu,'r-','LineWidth',2);
    
    xlabel(['$',labels(o),'$'],'interpreter','latex','FontSize',12);
    ylabel('$z$','interpreter','latex','FontSize',12);
    title('Reference Model','interpreter','latex','FontSize',12);
    
    axis tight
    ax2 = gca;
    
    % equlize axes
    YLim = [min(ax1.YLim(1),ax2.YLim(1)) max(ax1.YLim(2),ax2.YLim(2))];
    
    ax1.YLim = YLim;
    ax2.YLim = YLim;
    ax1.XLim = [min(Xo) max(Xo)];
    ax2.XLim = [min(Xo) max(Xo)];
    
end


