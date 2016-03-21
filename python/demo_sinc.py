import GPz
from numpy import *
import matplotlib.pyplot as plt
from numpy.linalg import *

########### Model options ###########

method = 'VL'               # select method, options = GL, VL, GD, VD, GC and VC
m = 200                     # number of basis functions to use

########### Generate Data ###########

n = 4000

X = linspace(-10, 10, n)
Xs = X.reshape(n, 1)
X = X[(X<-6)|(X>-3)]

n = len(X)
X = X.reshape(n, 1)

f_noise = (0.01+3 * sin(X) * (1 + exp(-0.1 * X)) ** -1) ** 2
Y = 10*sinc(2*X) + random.randn(n, 1) * f_noise


########### Start Script ###########

# optain an initial model using the default options
model = GPz.GP(m,method=method)

# train the model using the default options
model.train(X,Y)

# use the model to generate predictions for the test set
mu,sigma,variance,noise,PHI = model.predict(Xs)

########### Display Results ###########

plt.fill_between(Xs[:,0], mu[:,0]-2 * sqrt(sigma[:,0]), mu[:,0]+2 * sqrt(sigma[:,0]),facecolor=(0.85, 0.85, 0.85))
plt.plot(X, Y, 'b.')

SIGMAi = model.SIGMAi
muY = model.muY
w = model.w

[U,S,V] = svd(SIGMAi)
R = dot(U,diag(sqrt(S)))

k = 50

ws = dot(R,random.randn(len(w),k))+w
mus = dot(PHI,ws)+muY
plt.plot(Xs,mus)
plt.plot(Xs, mu, 'r-',linewidth=2)
plt.show()