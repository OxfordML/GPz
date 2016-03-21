from sklearn.decomposition import PCA
from scipy.optimize import fmin_bfgs
from numpy.linalg import inv,cholesky
from numpy import *
import matplotlib.pyplot as plt
import time


def sample(n, trainSample, validSample, testSample):

    if(trainSample<=1):
        validSample = ceil(n * validSample)
        testSample = ceil(n * testSample)
        trainSample = min([ceil(n * trainSample), n - testSample - validSample])

    r = random.permutation(n)

    validation = zeros(n, dtype=bool)
    testing = zeros(n, dtype=bool)
    training = zeros(n, dtype=bool)

    validation[r[0:validSample]] = True
    testing[r[validSample:validSample + testSample]] = True
    training[r[validSample + testSample:validSample + testSample + trainSample]] = True

    return training, validation, testing


def getOmega(y, method='normal', binWidth=0):

    n = len(y)

    if method == 'balanced':

        minY = min(y)
        maxY = max(y)

        if(binWidth==0):
            binWidth = (maxY-minY)/100

        bins = ceil((maxY-minY)/binWidth)
        centers = minY+[i*binWidth for i in range(0, bins+1)]
        h = plt.hist(y,centers)
        counts = h[0]
        D = Dxy(y,centers[0:int(bins)].reshape(bins,1)+binWidth/2)

        ind = D.argmin(1)
        omega = max(counts)/counts[ind]
        return omega.reshape(n, 1)
    elif method == 'normalized':
        return (y + 1) ** -2
    else:
        return ones((n, 1))

def metrics(y,mu,sigma,f):
    n = len(y)
    order = argsort(sigma,0)
    y = y[order]
    mu = mu[order]
    sigma = sigma[order]

    scores = cumsum(f(y,mu,sigma))
    scores = array([scores[i]/(i+1) for i in range(n)])

    return scores

def bin(x,y,bins):

    centers = linspace(min(x),max(x),bins)

    D = Dxy(x,centers.reshape(bins,1))
    ind = D.argmin(1)

    counts = zeros(bins)
    means = zeros(bins)
    stds = zeros(bins)

    for i in range(bins):
        counts[i] = sum(ind==i)
        if(counts[i]>0):
            means[i] = mean(y[ind==i])
            stds[i] = std(y[ind==i])

    keep = counts>0

    centers = centers[keep]
    means = means[keep]
    stds = stds[keep]

    return centers,means,stds

def Dxy(X, Y):
    x2 = sum(power(X, 2), 1)
    y2 = sum(power(Y, 2), 1).T
    xy = dot(X, Y.T)
    D = add(add(y2, -2 * xy).T, x2).T

    return D

class GP:

    def __init__(self, m,method='GL', joint=True, heteroscedastic=True,decorrelate=False):

        self.m = m
        self.method = method
        self.joint = joint
        self.heteroscedastic = heteroscedastic
        self.decorrelate = decorrelate
        self.pca = PCA(whiten=True)
        self.not_initalized = True

    def init(self,X,Y, omega=None, training=None):

        m = self.m
        method = self.method
        joint = self.joint
        heteroscedastic = self.heteroscedastic
        decorrelate = self.decorrelate
        pca = self.pca

        n,d = X.shape

        Xt = X

        if training is None:
            training = ones(n, dtype=bool)

        if omega is None:
            omega = ones((n, 1))

        if decorrelate:
            pca.fit(Xt[training, :])
            Xt = pca.transform(Xt)


        self.muY = mean(Y[training, :])

        Yt = Y-self.muY

        pca4k = PCA(whiten=True)
        pca4k.fit(Xt[training, :])
        P = pca4k.inverse_transform((random.rand(m, d) - 0.5)*sqrt(12))

        gamma = sqrt(2*power(m,1./d) / mean(Dxy(X[training, :], P),0))

        b = -log(var(Yt[training]))

        if joint:
            a_dim = m+d+1
        else:
            a_dim = m

        lnAlpha = -b*zeros(a_dim)

        if method == 'GL':
            GAMMA = mean(gamma)
        elif method == 'VL':
            GAMMA = gamma
        elif method == 'GD':
            GAMMA = ones((1,d))*mean(gamma)
        elif method == 'VD':
            GAMMA = ones((d,m))*gamma
        elif method == 'GC':
            GAMMA = eye(d) * mean(gamma)
        else:
            GAMMA = zeros((d, d, m))
            for j in range(m):
                GAMMA[:, :, j] = eye(d) * gamma[j]

        theta = concatenate([P.flatten(), GAMMA.flatten(), lnAlpha.flatten(), b.flatten()])

        if(heteroscedastic):

            theta = concatenate([theta.flatten(), zeros(2*m)])

            w,SIGMAi = self.GP_fun(theta, Xt, Yt, omega=omega,training=training, validation=[], returnModel=True)
            PHI,_ = self.GP_fun(theta, Xt, [],training=training)

            target = -log((Yt[training,:]-dot(PHI,w))**2)-b
            lnEta = log(var(target))*ones(m)

            # u = dot(inv(dot(PHI[:,0:m].T,PHI[:,0:m])+diag(exp(lnEta))),dot(PHI[:,0:m].T,target))
            theta = concatenate([P.flatten(), GAMMA.flatten(), lnAlpha.flatten(), b.flatten(), zeros(m), lnEta.flatten()])

        self.w = zeros((a_dim,1))
        self.SIGMAi = diag(exp(-lnAlpha))

        self.theta = theta
        self.not_initalized = False
        self.attempts = 0
        self.maxAttempts=inf

    def train(self, X, Y, omega=None, training=None, validation=None, maxIter=200,maxAttempts=inf):

        self.maxAttempts = maxAttempts
        self.attempts = 0

        n,d = X.shape

        if training is None:
            training = ones(n, dtype=bool)

        if omega is None:
            omega = ones((n, 1))

        if(self.not_initalized):
            self.init(X,Y,omega,training)

        if self.decorrelate:
            Xt = self.pca.transform(X)
        else:
            Xt = X

        Yt = Y-self.muY

        fx = self.GP_fun
        gx = self.GP_grad

        self.iter = 1
        self.timer = time.time()
        self.validLL = None

        self.theta = fmin_bfgs(fx, self.theta, fprime=gx, args=(Xt,Yt, omega, training, validation),callback=self.callbackF, disp=False, maxiter=maxIter)

        w, SIGMAi = fx(self.theta, Xt, Yt, omega,training, [], True)

        self.w = w
        self.SIGMAi = SIGMAi

        w, SIGMAi = fx(self.best_theta,Xt, Yt, omega,training, [], True)

        self.best_w = w
        self.best_SIGMAi = SIGMAi

    def predict(self,X,model='best'):

        n, d = X.shape

        if(model=='best'):
            theta = self.best_theta
            w = self.best_w
            SIGMAi = self.best_SIGMAi
        else:
            theta = self.theta
            w = self.w
            SIGMAi = self.SIGMAi


        if self.decorrelate:
            Xt = self.pca.transform(X)
        else:
            Xt = X

        PHI,lnBeta = self.GP_fun(theta, Xt, [])

        mu = dot(PHI, w) + self.muY

        modelV = sum(dot(PHI, SIGMAi) * PHI, 1).reshape(n, 1)
        noiseV = exp(-lnBeta)
        sigma = modelV+noiseV

        return mu,sigma,modelV, noiseV,PHI

    def GP_fun(self,theta, X, Y, omega=None, training=None, validation=None, returnModel=False):

        m = self.m
        joint = self.joint
        heteroscedastic = self.heteroscedastic
        method = self.method


        n,d = X.shape

        if training is None:
            training = ones(n, dtype=bool)

        if omega is None:
            omega = ones((n, 1))

        n = sum(training)

        if(joint):
            a_dim = m+d+1
        else:
            a_dim = m

        P = theta[0:m * d].reshape(m, d)

        GAMMA,lnPHI,g_dim,_ = self.getGAMMA_lnPHI(theta,X[training,:],P,m)

        PHI = exp(lnPHI)

        lnAlpha = theta[m*d+g_dim:m*d+g_dim+a_dim].reshape(a_dim,1)
        b = theta[m*d+g_dim+a_dim]

        alpha = exp(lnAlpha)

        lnBeta = b+log(omega[training])

        if(heteroscedastic):
            u = theta[m*d+g_dim+a_dim+1:m*d+g_dim+a_dim+1+m].reshape(m, 1)
            lnBeta = dot(PHI, u)+lnBeta

        beta = exp(lnBeta)

        if (joint):
            PHI = hstack([PHI, X[training, :], ones((n, 1))])


        if Y==[]:
            return PHI,lnBeta

        A = diag(alpha[:,0])

        PSi = PHI*beta
        PSiY = dot(PSi.T, Y[training, :])

        SIGMA = dot(PSi.T, PHI) + A

        Li = inv(cholesky(SIGMA))

        SIGMAi = dot(Li.T,Li)

        logdet = -2*sum(log(diag(Li)))

        w = dot(SIGMAi, PSiY)

        pred = dot(PHI, w)
        delta = pred - Y[training, :]

        nlogML = -0.5 * sum(delta ** 2 * beta) + 0.5 * sum(lnBeta) - 0.5 * sum(w ** 2 * alpha) + 0.5 * sum(lnAlpha) - 0.5 * logdet - (n / 2) * log(2 * pi)

        if(heteroscedastic):
            lnEta = theta[m*d+g_dim+a_dim+1+m:m*d+g_dim+a_dim+1+m+m].reshape(m, 1)
            eta = exp(lnEta)
            nlogML = nlogML - 0.5 * sum(u ** 2 * eta) + 0.5 * sum(lnEta)


        nlogML = -nlogML/n

        self.nlogML = nlogML

        if returnModel:
            return w, SIGMAi
        else:

            variance = sum(dot(PHI,SIGMAi) * PHI, 1).reshape(n, 1)
            sigma = variance+exp(-lnBeta)

            self.trainRMSE = sqrt(mean(omega[training] * delta ** 2))
            self.trainLL = mean(-0.5 * delta ** 2/sigma - 0.5 * log(sigma))-0.5*log(2*pi)

            if validation is not None:
                n = sum(validation)

                GAMMA,lnPHI,g_dim,_ = self.getGAMMA_lnPHI(theta,X[validation,:],P,m)

                PHI = exp(lnPHI)

                lnBeta = b+log(omega[validation])

                if(heteroscedastic):
                    lnBeta = dot(PHI, u)+lnBeta

                if (joint):
                    PHI = hstack([PHI, X[validation, :], ones((n, 1))])

                variance = sum(dot(PHI,SIGMAi) * PHI, 1).reshape(n, 1)
                sigma = variance+exp(-lnBeta)

                delta = dot(PHI, w) - Y[validation, :]

                self.validRMSE = sqrt(mean(omega[validation] * delta ** 2))
                self.validLL = mean(-0.5 * delta ** 2/sigma - 0.5 * log(sigma))-0.5*log(2*pi)

            return nlogML

    def GP_grad(self,theta, X, Y, omega=None, training=None, validation=None):

        m = self.m
        joint = self.joint
        heteroscedastic = self.heteroscedastic
        method = self.method

        n, d = X.shape

        if training is None:
            training = ones(n, dtype=bool)

        if omega is None:
            omega = ones((n, 1))

        n,d = X[training, :].shape

        if(joint):
            a_dim = m+d+1
        else:
            a_dim = m

        P = theta[0:m * d].reshape(m, d)

        GAMMA,lnPHI,g_dim,dGAMMA = self.getGAMMA_lnPHI(theta,X[training,:],P,m)

        PHI = exp(lnPHI)

        lnAlpha = theta[m*d+g_dim:m*d+g_dim+a_dim].reshape(a_dim, 1)
        b = theta[m*d+g_dim+a_dim]

        alpha = exp(lnAlpha)

        lnBeta = b+log(omega[training])

        if(heteroscedastic):
            u = theta[m*d+g_dim+a_dim+1:m*d+g_dim+a_dim+1+m].reshape(m, 1)
            lnBeta = lnBeta+dot(PHI, u)

        beta = exp(lnBeta)

        if (joint):
            PHI = hstack([PHI, X[training, :], ones((n, 1))])

        A = diag(alpha[:, 0])

        PSi = PHI * beta
        PSiY = dot(PSi.T, Y[training])

        SIGMA = dot(PSi.T, PHI) + A
        Li = inv(cholesky(SIGMA))

        SIGMAi = dot(Li.T,Li)

        w = dot(SIGMAi, PSiY)

        delta = dot(PHI, w) - Y[training]

        dwda = -dot(SIGMAi, alpha * w)

        dlnAlpha = -sum(dot(dwda, (beta * delta).T) * PHI.T, 1).reshape(alpha.shape) - alpha * w * dwda - 0.5*alpha * w ** 2 -0.5*diag(SIGMAi).reshape(alpha.shape) * alpha + 0.5

        variance = sum(dot(PHI, SIGMAi) * PHI, 1).reshape(n, 1)
        dbeta = -0.5 * (variance + delta ** 2) * beta + 0.5
        db = sum(dbeta)

        if(heteroscedastic):
            lnEta = theta[m*d+g_dim+a_dim+1+m:m*d+g_dim+a_dim+1+m+m].reshape(m, 1)
            eta = exp(lnEta)
            du = dot(PHI[:, 0:m].T, dbeta)-u*eta
            dlnEta = -0.5*eta*u**2+0.5
            dlnPHI = (dot(dbeta, u.T) - dot(delta * beta, w[0:m].T) - dot(PSi, SIGMAi[:, 0:m])) * PHI[:, 0:m]
        else:
            dlnPHI = (- dot(delta * beta, w[0:m].T) - dot(PSi, SIGMAi[:, 0:m])) * PHI[:, 0:m]

        dP = zeros((m, d))

        for j in range(m):
            Delta = X[training, :] - P[j, :]
            if method=='GL':
                dP[j, :] = sum(Delta * dlnPHI[:, j].reshape(n, 1), 0) * GAMMA ** 2
                dGAMMA = dGAMMA+2*sum(dlnPHI[:,j]* lnPHI[:,j])*GAMMA**-1
            elif method=='VL':
                dP[j, :] = sum(Delta * dlnPHI[:, j].reshape(n, 1), 0) * GAMMA[j] ** 2
                dGAMMA[j] = 2*sum(dlnPHI[:,j]* lnPHI[:,j])*GAMMA[j]**-1
            elif method=='GD':
                dP[j, :] = dot(dlnPHI[:,j].T,Delta)*GAMMA**2
                dGAMMA = dGAMMA-diag(dot(GAMMA.reshape(d,1)*(Delta*dlnPHI[:,j].reshape(n, 1)).T,Delta))
            elif method=='VD':
                dP[j, :] = dot(dlnPHI[:,j].T,Delta)*GAMMA[j,:]**2
                dGAMMA[j,:] = -diag(dot(GAMMA[j,:].reshape(d,1)*(Delta*dlnPHI[:,j].reshape(n, 1)).T,Delta))
            elif method=='GC':
                dP[j, :] = dot(dot(dlnPHI[:,j].T,Delta),dot(GAMMA.T,GAMMA))
                dGAMMA = dGAMMA-dot(dot(GAMMA,(Delta*dlnPHI[:,j].reshape(n, 1)).T),Delta)
            else:
                dP[j, :] = dot(dot(dlnPHI[:,j].T,Delta),dot(GAMMA[:,:,j].T,GAMMA[:,:,j]))
                dGAMMA[:,:,j] = -dot(dot(GAMMA[:,:,j],(Delta*dlnPHI[:,j].reshape(n, 1)).T),Delta)


        if(heteroscedastic):
            grad = concatenate((dP.flatten(), dGAMMA.flatten(), dlnAlpha.flatten(), db.flatten(), du.flatten(),dlnEta.flatten()))
        else:
            grad = concatenate((dP.flatten(), dGAMMA.flatten(), dlnAlpha.flatten(), db.flatten()))

        if(self.attempts>=self.maxAttempts):
            grad = 0*grad

        return -grad/n

    def getGAMMA_lnPHI(self,theta,X,P,m):

        method = self.method

        n,d = X.shape

        if(method=='GL'):
            g_dim = 1
            dGAMMA = 0
            GAMMA = theta[m*d]
            lnPHI = -0.5 * Dxy(X, P) * GAMMA ** 2


        elif(method=='VL'):
            g_dim = m
            dGAMMA = zeros(m)
            GAMMA = theta[m*d:m*d+m]
            lnPHI = -0.5 * Dxy(X, P) * GAMMA ** 2
        elif(method=='GD'):
            g_dim = d
            dGAMMA = zeros((1, d))
            GAMMA = theta[m*d:m*d+d].reshape(1, d)

            lnPHI = zeros((n, m))
            for j in range(m):
                Delta = X - P[j, :]
                lnPHI[:, j] = -0.5 * sum(power(Delta*GAMMA, 2), 1)
        elif(method=='VD'):
            g_dim = m*d
            GAMMA = theta[m*d:m*d+m*d].reshape(m, d)
            dGAMMA = zeros((m, d))
            lnPHI = zeros((n, m))
            for j in range(m):
                Delta = X - P[j, :]
                lnPHI[:, j] = -0.5 * sum(power(Delta*GAMMA[j,:], 2), 1)
        elif(method=='GC'):
            g_dim = d*d
            dGAMMA = zeros((d, d))
            GAMMA = theta[m*d:m*d+d*d].reshape(d, d)

            lnPHI = zeros((n, m))

            for j in range(m):
                Delta = X - P[j, :]
                lnPHI[:, j] = -0.5 * sum(power(dot(Delta, GAMMA.T), 2), 1)
        else:
            g_dim = d*d*m
            dGAMMA = zeros((d, d, m))
            GAMMA = theta[m*d:m*d+d*d*m].reshape(d, d, m)

            lnPHI = zeros((n, m))

            for j in range(m):
                Delta = X - P[j, :]
                lnPHI[:, j] = -0.5 * sum(power(dot(Delta, GAMMA[:,:,j].T), 2), 1)

        return GAMMA,lnPHI,g_dim,dGAMMA

    def callbackF(self,theta):

        if (self.iter==1):
            if self.validLL is None:
                print  '{0:4s}\t{1:9s}\t\t{2:9s}\t\t{3:9s}\t\t{4:9s}'.format('Iter', ' logML/n', ' trainRMSE', ' trainLL', ' Time')
            else:
                print  '{0:4s}\t{1:9s}\t\t{2:9s}\t\t{3:9s}\t\t{4:9s}\t\t{5:9s}\t\t{6:9s}'.format('Iter', ' logML/n', ' trainRMSE',
                                                                                   ' trainRMSE/n', ' validRMSE', ' validLL', ' Time')
        if self.validLL is None:
            print '{0:4d}\t{1: 1.7e}\t{2: 1.7e}\t{3: 1.7e}\t{4: 1.7e}'.format(self.iter, -self.nlogML, self.trainRMSE,self.trainLL,time.time()-self.timer)
            self.best_theta = theta
            self.best_valid = self.trainLL
        else:

            if self.iter==1 or self.validLL >= self.best_valid:
                print '{0:4d}\t{1: 1.7e}\t{2: 1.7e}\t{3: 1.7e}\t{4: 1.7e}\t[{5: 1.7e}]\t{6: 1.7e}'.format(self.iter, -self.nlogML, self.trainRMSE, self.trainLL, self.validRMSE, self.validLL,time.time()-self.timer)
                self.best_theta = theta
                self.best_valid = self.validLL
                self.attempts = 0
            else:
                print '{0:4d}\t{1: 1.7e}\t{2: 1.7e}\t{3: 1.7e}\t{4: 1.7e}\t {5: 1.7e} \t{6: 1.7e}'.format(self.iter, -self.nlogML, self.trainRMSE, self.trainLL, self.validRMSE, self.validLL,time.time()-self.timer)
                self.attempts = self.attempts+1

        if(self.attempts>self.maxAttempts):
            print 'No improvment after maximum number of attempts'
        self.timer = time.time()
        self.iter += 1