
addpath GPz/

addpath(genpath('minFunc_2012/'))    % path to minfunc
%%%%%%%%%%%%%% Model options %%%%%%%%%%%%%%%%

method = 'VC';              % select method, options = GL, VL, GD, VD, GC and VC [required]

m = 25;                     % number of basis functions to use [required]

joint = true;               % jointly learn a prior linear mean function [default=true]

heteroscedastic = true;     % learn a heteroscedastic noise process, set to false interested only in point estimates

csl_method = 'normal';      % cost-sensitive learning option: [default='normal']
                            %       'balanced':     to weigh rare samples more heavly during train
                            %       'normalized':   assigns an error cost for each sample = 1/(z+1)
                            %       'normal':       no weights assigned, all samples are equally important
                            
binWidth = 0.1;             % the width of the bin for 'balanced' cost-sensitive learning [default=range(z_spec)/100]

decorrelate = true;         % preprocess the data using PCA [default=true]

%%%%%%%%%%%%%% Training options %%%%%%%%%%%%%%%%  
dataPath = '../data/sdss_sample.csv';   % path to the data set, has to be in the following format m_1,m_2,..,m_k,e_1,e_2,...,e_k,z_spec
                                % where m_i is the i-th magnitude, e_i is its associated uncertainty and z_spec is the spectroscopic redshift
                                % [required]
                                
maxIter = 500;                  % maximum number of iterations [default=200]
maxAttempts = 50;               % maximum iterations to attempt if there is no progress on the validation set [default=infinity]


trainSplit = 0.2;              % percentage of data to use for training
validSplit = 0.2;              % percentage of data to use for validation
testSplit  = 0.6;              % percentage of data to use for testing
%%%%%%%%%%%%%% Start of script %%%%%%%%%%%%%%%% 


% read data from file
X = csvread(dataPath);
Y = X(:,end);
X(:,end) = [];

[n,d] = size(X);
filters = d/2;

% log the uncertainties of the magnitudes, any additional preprocessing should be placed here
X(:,filters+1:end) = log(X(:,filters+1:end));

% sample training, validation and testing sets from the data
[training,validation,testing] = sample(n,trainSplit,validSplit,testSplit); 

% you can also select the size of each sample
% [training,validation,testing] = sample(n,10000,10000,10000);

% get the weights for cost-sensitive learning
omega = getOmega(Y,csl_method,binWidth); 


% initialize the initial model
model = init(X,Y,method,m,'omega',omega,'training',training,'heteroscedastic',heteroscedastic,'joint',joint,'decorrelate',decorrelate);

% train the model
model = train(model,X,Y,'omega',omega,'training',training,'validation',validation,'maxIter',maxIter,'maxAttempts',maxAttempts);

%%%%%%%% NOTE %%%%%%%
% you can train the model again, even using different data, by executing:
% model = train(model,X,Y,options);

% use the model to generate predictions for the test set
[mu,sigma,modelV,noiseV] = predict(X(testing,:),model);


%%%%%%%%%%%%%% Display Results %%%%%%%%%%%%%%%% 

% compute metrics 
rmse = sqrt(metrics(Y(testing),mu,sigma,@(y,mu,sigma) (y-mu).^2));
mll = metrics(Y(testing),mu,sigma,@(y,mu,sigma) -0.5*(y-mu).^2./sigma - 0.5*log(sigma)-0.5*log(2*pi));
nmad15 = metrics(Y(testing),mu,sigma,@(y,mu,sigma) 100*(abs(y-mu)./(y+1)<0.15));
nmad05 = metrics(Y(testing),mu,sigma,@(y,mu,sigma) 100*(abs(y-mu)./(y+1)<0.05));
bias = metrics(Y(testing),mu,sigma,@(y,mu,sigma) y-mu);

% print metrics for the entire data
fprintf('RMSE\t\tMLL\t\tNMAD15\t\tNMAD05\t\tBIAS\n')
fprintf('%f\t%f\t%f\t%f\t%f\n',rmse(end),mll(end),nmad15(end),nmad05(end),bias(end))

% plot scatter plots for density and uncertainty
[x,y,color,counts]=reduce(Y(testing),mu,sigma,200);

figure;scatter(x,y,12,log(color),'s','filled');title('Uncertainty');xlabel('Spectroscopic Redshift');ylabel('Photometric Redshift');
figure;scatter(x,y,12,log(counts),'s','filled');title('Density');xlabel('Spectroscopic Redshift');ylabel('Photometric Redshift');

% plot the change in metrics as functions of data percentage
x = [1 5:5:100];
ind = round(x*length(rmse)/100);

figure;plot(x,rmse(ind),'o-');xlabel('Percentage of Data');ylabel('RMSE');
figure;plot(x,mll(ind),'o-');xlabel('Percentage of Data');ylabel('MLL');
figure;plot(x,nmad05(ind),'o-');xlabel('Percentage of Data');ylabel('NMAD05');
figure;plot(x,nmad15(ind),'o-');xlabel('Percentage of Data');ylabel('NMAD15');
figure;plot(x,bias(ind),'o-');xlabel('Percentage of Data');ylabel('BIAS');

% plot mean and standard deviation of different scores as functions of spectroscopic redshift using 20 bins
[centers,means,stds] = bin(Y(testing),Y(testing)-mu,20);
figure;errorbar(centers,means,stds,'s');xlabel('Spectroscopic Redshift');ylabel('Bias');

[centers,means,stds] = bin(Y(testing),sqrt(modelV),20);
figure;errorbar(centers,means,stds,'s');xlabel('Spectroscopic Redshift');ylabel('Model Uncertainty');

[centers,means,stds] = bin(Y(testing),sqrt(noiseV),20);
figure;errorbar(centers,means,stds,'s');xlabel('Spectroscopic Redshift');ylabel('Noise Uncertainty');

% save output as comma seperated values (mean,sigma,model_variance,noise_variance)
csvwrite([method,'_',num2str(m),'_',csl_method,'.csv'],[Y(testing) mu sigma modelV noiseV]);