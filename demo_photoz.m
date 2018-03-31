rng(1); % fix random seed
addpath GPz/ % path to GPz
addpath(genpath('minFunc_2012/'))       % path to minfunc
 
%%%%%%%%%%%%%% Model options %%%%%%%%%%%%%%%%
 
m = 100;                                % number of basis functions to use [required]
 
method = 'VD';                          % select a method, options = GL, VL, GD, VD, GC and VC [required]
 
 
heteroscedastic = true;                 % learn a heteroscedastic noise process, set to false if only interested in point estimates [default=true]
 
normalize = true;                       % pre-process the input by subtracting the means and dividing by the standard deviations [default=true]

maxIter = 500;                          % maximum number of iterations [default=200]
maxAttempts = 50;                       % maximum iterations to attempt if there is no progress on the validation set [default=infinity]
 
 
trainSplit = 0.2;                       % percentage of data to use for training
validSplit = 0.2;                       % percentage of data to use for validation
testSplit  = 0.6;                       % percentage of data to use for testing

inputNoise = true;                      % false = use mag errors as additional inputs, true = use mag errors as additional input noise

csl_method = 'normal';                  % cost-sensitive learning option: [default='normal']
                                        %       'balanced':     to weigh
                                        %       rare samples more heavily during training
                                        %       'normalized':   assigns an error cost for each sample = 1/(z+1)
                                        %       'normal':       no weights assigned, all samples are equally important
 
binWidth = 0.1;                         % the width of the bin for 'balanced' cost-sensitive learning [default=range(output)/100]
%%%%%%%%%%%%%% Prepare data %%%%%%%%%%%%%% 
 
dataPath = 'data/sdss_sample.csv';   % path to the data set, has to be in the following format m_1,m_2,..,m_k,e_1,e_2,...,e_k,z_spec
                                        % where m_i is the i-th magnitude, e_i is its associated uncertainty and z_spec is the spectroscopic redshift
                                        % [required]
outPath = [];                           % if set to a path, the output will be saved to a csv file.
                                        
% read data from file
X = csvread(dataPath);
Y = X(:,end);
X(:,end) = [];
 
[n,d] = size(X);
filters = d/2;
 
% you can also select the size of each sample
% [training,validation,testing] = sample(n,10000,10000,10000);
 
% get the weights for cost-sensitive learning
omega = getOmega(Y,csl_method,binWidth); 

if(inputNoise)
    % treat the mag-errors as input noise variance
    Psi = X(:,filters+1:end).^2;
    X(:,filters+1:end) = [];
else
    % treat the mag-errors as input additional inputs
    X(:,filters+1:end) = log(X(:,filters+1:end));
    Psi = [];
end

% select training, validation and testing sets from the data
[training,validation,testing] = sample(n,trainSplit,validSplit,testSplit); 

%%%%%%%%%%%%%% Fit the model %%%%%%%%%%%%%%

% initialize the model
model = init(X,Y,method,m,'omega',omega,'training',training,'heteroscedastic',heteroscedastic,'normalize',normalize,'Psi',Psi);
% train the model
model = train(model,X,Y,'omega',omega,'training',training,'validation',validation,'maxIter',maxIter,'maxAttempts',maxAttempts,'Psi',Psi); 


%%%%%%%%%%%%%% Compute Metrics %%%%%%%%%%%%%%

% use the model to generate predictions for the test set
[mu,sigma,nu,beta_i,gamma] = predict(X,model,'Psi',Psi,'selection',testing);

% mu     = the best point estimate
% nu     = variance due to data density
% beta_i = variance due to output noise
% gamma  = variance due to input noise
% sigma  = nu+beta_i+gamma

% compute metrics 
 
%root mean squared error, i.e. sqrt(mean(errors^2))
rmse = sqrt(metrics(Y(testing),mu,sigma,@(y,mu,sigma) (y-mu).^2)); 
 
% mean log likelihood mean(-0.5*errors^2/sigma -0.5*log(sigma)-0.5*log(2*pi))
mll = metrics(Y(testing),mu,sigma,@(y,mu,sigma) -0.5*(y-mu).^2./sigma - 0.5*log(sigma)-0.5*log(2*pi));
 
% fraction of data where |z_spec-z_phot|/(1+z_spec)<0.15
fr15 = metrics(Y(testing),mu,sigma,@(y,mu,sigma) 100*(abs(y-mu)./(y+1)<0.15));
 
% fraction of data where |z_spec-z_phot|/(1+z_spec)<0.05
fr05 = metrics(Y(testing),mu,sigma,@(y,mu,sigma) 100*(abs(y-mu)./(y+1)<0.05));
 
% bias, i.e. mean(errors)
bias = metrics(Y(testing),mu,sigma,@(y,mu,sigma) y-mu);
 
% print metrics for the entire data
fprintf('RMSE\t\tMLL\t\tFR15\t\tFR05\t\tBIAS\n')
fprintf('%f\t%f\t%f\t%f\t%f\n',rmse(end),mll(end),fr15(end),fr05(end),bias(end))
 
%%%%%%%%%%%%%% Display Results %%%%%%%%%%%%%%%% 
 
% reduce the sample for efficient plotting
[x,y,color,counts]=reduce(Y(testing),mu,sigma,200);
 
figure;scatter(x,y,12,log(color),'s','filled');title('Uncertainty');xlabel('Spectroscopic Redshift');ylabel('Photometric Redshift');colormap jet;
figure;scatter(x,y,12,log(counts),'s','filled');title('Density');xlabel('Spectroscopic Redshift');ylabel('Photometric Redshift');colormap jet;
 
% plot the change in metrics as functions of data percentage
x = [1 5:5:100];
ind = round(x*length(rmse)/100);
 
figure;plot(x,rmse(ind),'o-');xlabel('Percentage of Data');ylabel('RMSE');
figure;plot(x,mll(ind),'o-');xlabel('Percentage of Data');ylabel('MLL');
figure;plot(x,fr05(ind),'o-');xlabel('Percentage of Data');ylabel('FR05');
figure;plot(x,fr15(ind),'o-');xlabel('Percentage of Data');ylabel('FR15');
figure;plot(x,bias(ind),'o-');xlabel('Percentage of Data');ylabel('BIAS');
 
% plot mean and standard deviation of different scores as functions of spectroscopic redshift using 20 bins
[centers,means,stds] = bin(Y(testing),Y(testing)-mu,20);
figure;errorbar(centers,means,stds,'s');xlabel('Spectroscopic Redshift');ylabel('Bias');
 
[centers,means,stds] = bin(Y(testing),sqrt(nu),20);
figure;errorbar(centers,means,stds,'s');xlabel('Spectroscopic Redshift');ylabel('Model Uncertainty');
 
[centers,means,stds] = bin(Y(testing),sqrt(beta_i),20);
figure;errorbar(centers,means,stds,'s');xlabel('Spectroscopic Redshift');ylabel('Noise Uncertainty');
 
% save output as comma separated values
if(~isempty(outPath))
    csvwrite([method,'_',num2str(m),'_',csl_method,'.csv'],[Y(testing) mu sigma nu beta_i gamma]);
end
