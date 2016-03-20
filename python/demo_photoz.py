from numpy import *
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import GPz

########### Model options ###############

method = 'VC'               # select method, options = GL, VL, GD, VD, GC and VC [required]
                            #
m = 25                      # number of basis functions to use [required]
                            #
joint = True                # jointly learn a prior linear mean function [default=true]
                            #
heteroscedastic = True      # learn a heteroscedastic noise process, set to false interested only in point estimates
                            #
csl_method = 'normal'       # cost-sensitive learning option: [default='normal']
                            #       'balanced':     to weigh rare samples more heavly during train
                            #       'normalized':   assigns an error cost for each sample = 1/(z+1)
                            #       'normal':       no weights assigned, all samples are equally important
                            #
binWidth = 0.1              # the width of the bin for 'balanced' cost-sensitive learning [default=range(z_spec)/100]

decorrelate = True          # preprocess the data using PCA [default=False]

########### Training options ###########

dataPath = '../data/sdss_sample.csv'    # path to the data set, has to be in the following format m_1,m_2,..,m_k,e_1,e_2,...,e_k,z_spec
                                        # where m_i is the i-th magnitude, e_i is its associated uncertainty and z_spec is the spectroscopic redshift
                                        # [required]

maxIter = 500                  # maximum number of iterations [default=200]

trainSplit = 0.2               # percentage of data to use for training
validSplit = 0.2               # percentage of data to use for validation
testSplit  = 0.6               # percentage of data to use for testing

########### Start of script ###########

# read data from file
data = loadtxt(open(dataPath,"rb"),delimiter=",")

n = len(data)

X = data[:, 0:10]
Y = data[:, 10].reshape(n, 1)

n,d = X.shape

filters = d/2

# log the uncertainties of the magnitudes, any additional preprocessing should be placed here
X[:, filters:] = log(X[:, filters:])

# sample training, validation and testing sets from the data
training,validation,testing = GPz.split(n,trainSplit,validSplit,testSplit);

# you can also sample by percentage for example
# training,validation,testing = GPz.sample(n,10000,10000,10000);

# get the weights for cost-sensitive learning
omega = GPz.getOmega(Y, method=csl_method)

# initialize the initial model
model = GPz.init(X.copy(), Y.copy(), method,m, omega=omega, training=training,heteroscedastic=heteroscedastic,joint=joint,decorrelate=decorrelate)

# train the model
model = GPz.train(model, X.copy(), Y.copy(), omega=omega, training=training, validation=validation, maxIter=maxIter)

########### NOTE ###########
# you can train the model gain, eve using different data, by executing:
# model = train(model,X,Y,options)

# use the model to generate predictions for the test set
mu,sigma,modelV,noiseV,_ = GPz.predict(X[testing,:].copy(), model)


########### Display Results ###########

# compute metrics
rmse = sqrt(GPz.metrics(Y[testing],mu,sigma,lambda y,mu,sigma: (y-mu)**2))
mll = GPz.metrics(Y[testing],mu,sigma,lambda y,mu,sigma: -0.5*(y-mu)**2/sigma-0.5*log(sigma)-0.5*log(2*pi))
nmad15 = GPz.metrics(Y[testing],mu,sigma,lambda y,mu,sigma: 100.0*(abs(y-mu)/(y+1.0)<0.15))
nmad05 = GPz.metrics(Y[testing],mu,sigma,lambda y,mu,sigma: 100.0*(abs(y-mu)/(y+1.0)<0.05))
bias = GPz.metrics(Y[testing],mu,sigma,lambda y,mu,sigma: y-mu)

# print metrics for the entire data
print '{0:4s}\t\t\t{1:3s}\t\t\t{2:6s}\t\t\t{3:6s}\t\t\t{4:4s}'.format('RMSE', ' MLL', ' NMAD15', ' NMAD05', ' BIAS')
print '{0:1.7e}\t{1: 1.7e}\t{2: 1.7e}\t{3: 1.7e}\t{4: 1.7e}'.format(rmse[-1], mll[-1], nmad15[-1],nmad05[-1],bias[-1])

# plot scatter plots for density and uncertainty
f = plt.figure(1)
plt.scatter(Y[testing,:],mu,s=5,c=log(squeeze(sigma)), edgecolor='')
f.show()

f = plt.figure(2)
xy = hstack([Y[testing,:],mu]).T
z = gaussian_kde(xy)(xy)
plt.scatter(Y[testing,:],mu,c=z,s=5, edgecolor='')
f.show()

# plot the change in metrics as functions of data percentage
x = array(range(0,20+1))*5
x[0]=1

ind = x*len(rmse)/100

f = plt.figure(3)
plt.plot(x,rmse[ind-1],'o-')
plt.xlabel('Percentage of Data')
plt.ylabel('RMSE')
f.show()

f = plt.figure(4)
plt.plot(x,mll[ind-1],'o-')
plt.xlabel('Percentage of Data')
plt.ylabel('MLL')
f.show()

f = plt.figure(5)
plt.plot(x,nmad15[ind-1],'o-')
plt.xlabel('Percentage of Data')
plt.ylabel('NMAD15')
f.show()

f = plt.figure(6)
plt.plot(x,nmad05[ind-1],'o-')
plt.xlabel('Percentage of Data')
plt.ylabel('NMAD05')
f.show()

f = plt.figure(7)
plt.plot(x,bias[ind-1],'o-')
plt.xlabel('Percentage of Data')
plt.ylabel('BIAS')
f.show()

# plot mean and standard deviation of different scores as functions of spectroscopic redshift using 20 bins
f = plt.figure(8)
centers,means,stds = GPz.bin(Y[testing],Y[testing]-mu,20)
plt.errorbar(centers,means,stds,fmt='o')
plt.xlabel('Spectroscopic Redshift')
plt.ylabel('Bias')
f.show()

f = plt.figure(9)
centers,means,stds = GPz.bin(Y[testing],sqrt(modelV),20)
plt.errorbar(centers,means,stds,fmt='o')
plt.xlabel('Spectroscopic Redshift')
plt.ylabel('Model Uncertainty')
f.show()

f = plt.figure(10)
centers,means,stds = GPz.bin(Y[testing],sqrt(noiseV),20)
plt.errorbar(centers,means,stds,fmt='o')
plt.xlabel('Spectroscopic Redshift')
plt.ylabel('Noise Uncertainty')
f.show()

# save output as a comma seperated values (mean,sigma,model_variance,noise_variance)
savetxt(method+'_'+str(m)+'_'+csl_method+'.csv', array([mu,sigma,modelV,noiseV])[:,:,0].T, delimiter=',')