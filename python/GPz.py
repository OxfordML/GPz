from sklearn.decomposition import PCA
from scipy.optimize import fmin_bfgs
from numpy.linalg import *
from numpy import *
import matplotlib.pyplot as plt


def init(X, y, method, m, omega=[], training=[], joint=True, heteroscedastic=True,decorrelate=False):

    n,d = X.shape


    if training==[]:
        training = ones(n, dtype=bool)

    if omega==[]:
        omega = ones((n, 1))

    pca = PCA(whiten=True)

    if decorrelate:
        pca.fit(X[training, :])
        X = pca.transform(X)

    muY = mean(y[training, :])

    y -= muY

    pca4k = PCA(whiten=True)
    pca4k.fit(X[training, :])
    P = pca4k.inverse_transform((random.rand(m, d) - 0.5)*sqrt(12))

    gamma = sqrt(2*power(m,1./d) / Dxy(X[training, :], P).mean(0))

    b = -log(var(y[training, :]))

    if joint:
        lnAlpha = -b*zeros((m + d + 1))
    else:
        lnAlpha = -b*zeros(m)

    if method == 'GL':
        fx = GL
        GAMMA = mean(gamma)
    elif method == 'VL':
        fx = VL
        GAMMA = gamma
    elif method == 'GD':
        fx = GD
        GAMMA = ones((1,d))*mean(gamma)
    elif method == 'VD':
        fx = VD
        GAMMA = ones((m,d))*gamma.reshape(m,1)
    elif method == 'GC':
        fx = GC
        GAMMA = eye(d) * mean(gamma)
    else:
        fx = VC
        GAMMA = zeros((d, d, m))

        for j in range(m):
            GAMMA[:, :, j] = eye(d) * gamma[j]

    theta = concatenate([P.flatten(), GAMMA.flatten(), lnAlpha.flatten(), b.flatten()])

    if(heteroscedastic):
        u = zeros(m)
        lnEta = zeros(m)
        theta = concatenate([theta.flatten(), u.flatten(), lnEta.flatten()])
        w,SIGMAi = fx(theta, m, X, y, omega=omega, joint=joint,heteroscedastic=heteroscedastic, training=training, validation=[], returnModel=True)
        PHI,_ = fx(theta, m, X, [], omega=omega, joint=joint,heteroscedastic=heteroscedastic, training=training, validation=[], returnModel=False)

        target = -log((y[training,:]-dot(PHI,w))**2)-b
        lnEta = log(var(target))*ones(m)

        # u = dot(inv(dot(PHI[:,0:m].T,PHI[:,0:m])+diag(exp(lnEta))),dot(PHI[:,0:m].T,target))
        theta = concatenate([P.flatten(), GAMMA.flatten(), lnAlpha.flatten(), b.flatten(), u.flatten(), lnEta.flatten()])

    w = zeros((len(lnAlpha),1))
    SIGMAi = diag(exp(-lnAlpha))


    model = (theta,m,w,SIGMAi, method, joint,heteroscedastic, decorrelate, muY, pca)

    return model


def train(model, X, y, omega=[], training=[], validation=[], maxIter=200):

    global validLL
    global iter

    n,d = X.shape

    theta, m, w, SIGMAi, method, joint,heteroscedastic, decorrelate, muY, pca = model

    if training==[]:
        training = ones(n, dtype=bool)

    if omega==[]:
        omega = ones((n, 1))

    if decorrelate:
        X = pca.transform(X)

    y -= muY

    if method == 'GL':
        fx = GL
        gx = GL_grad
    elif method == 'VL':
        fx = VL
        gx = VL_grad
    elif method == 'GD':
        fx = GD
        gx = GD_grad
    elif method == 'VD':
        fx = VD
        gx = VD_grad
    elif method == 'GC':
        fx = GC
        gx = GC_grad
    else:
        fx = VC
        gx = VC_grad

    theta = fmin_bfgs(fx, theta, fprime=gx, args=(m,X, y, omega, joint,heteroscedastic, training, validation), disp=False,callback=callbackF, maxiter=maxIter)

    global best_theta

    theta = best_theta

    w, SIGMAi = fx(theta, m, X, y, omega, joint,heteroscedastic, training, [], True)

    model = (theta, m, w, SIGMAi, method, joint,heteroscedastic, decorrelate, muY, pca)

    if 'validLL' in globals():
        del validLL

    del iter

    return model


def predict(X, model):

    theta, m, w, SIGMAi, method, joint,heteroscedastic, decorrelate, muY, pca = model

    n, d = X.shape

    if decorrelate:
        X = pca.transform(X)

    if method == 'GL':
        PHI,lnBeta = GL(theta, m, X, [], joint=joint,heteroscedastic=heteroscedastic)
    elif method == 'VL':
        PHI,lnBeta = VL(theta, m, X, [], joint=joint,heteroscedastic=heteroscedastic)
    elif method == 'GD':
        PHI,lnBeta = GD(theta, m, X, [], joint=joint,heteroscedastic=heteroscedastic)
    elif method == 'VD':
        PHI,lnBeta = VD(theta, m, X, [], joint=joint,heteroscedastic=heteroscedastic)
    elif method == 'GC':
        PHI,lnBeta = GC(theta, m, X, [], joint=joint,heteroscedastic=heteroscedastic)
    else:
        PHI,lnBeta = VC(theta, m, X, [], joint=joint,heteroscedastic=heteroscedastic)

    mu = dot(PHI, w) + muY

    modelV = sum(dot(PHI, SIGMAi) * PHI, 1).reshape(n, 1)
    noiseV = exp(-lnBeta)
    sigma = modelV+noiseV
    return mu,sigma,modelV, noiseV,PHI


def GL(theta, m, X, y, omega=[], joint=False,heteroscedastic=False, training=[], validation=[], returnModel=False):

    global nlogML

    global trainRMSE
    global trainLL

    global validRMSE
    global validLL


    n,d = X.shape
    if training==[]:
        training = ones(n, dtype=bool)

    if omega==[]:
        omega = ones((n, 1))

    n = sum(training)

    if(joint):
        a_dim = m+d+1
    else:
        a_dim = m

    P = theta[0:m * d].reshape(m, d)
    gamma = theta[m*d]
    lnAlpha = theta[m*d+1:m*d+1+a_dim].reshape(a_dim, 1)
    b = theta[m*d+1+a_dim]

    alpha = exp(lnAlpha)

    lnPHI = -0.5 * Dxy(X[training, :], P) * gamma ** 2
    PHI = exp(lnPHI)

    lnBeta = b+log(omega[training])

    if(heteroscedastic):
        u = theta[m*d+1+a_dim+1:m*d+1+a_dim+1+m].reshape(m, 1)
        lnBeta = dot(PHI, u)+lnBeta

    beta = exp(lnBeta)

    if (joint):
        PHI = hstack([PHI, X[training, :], ones((n, 1))])


    if y==[]:
        return PHI,lnBeta

    A = diag(alpha[:, 0])

    PSi = PHI * (beta)
    PSiY = dot(PSi.T, y[training, :])

    SIGMA = dot(PSi.T, PHI) + A

    Li = inv(cholesky(SIGMA))

    SIGMAi = dot(Li.T,Li)

    logdet = -2*sum(log(diag(Li)))

    w = dot(SIGMAi, PSiY)

    pred = dot(PHI, w)
    delta = pred - y[training, :]

    nlogML = -0.5 * sum(delta ** 2 * beta) + 0.5 * sum(lnBeta) - 0.5 * sum(w ** 2 * alpha) + 0.5 * sum(lnAlpha) - 0.5 * logdet - (n / 2) * log(2 * pi)

    if(heteroscedastic):
        lnEta = theta[m*d+1+a_dim+1+m:m*d+1+a_dim+1+m+m].reshape(m, 1)
        eta = exp(lnEta)
        nlogML = nlogML - 0.5 * sum(u ** 2 * eta) + 0.5 * sum(lnEta)


    nlogML = -nlogML/n

    if returnModel:
        return w, SIGMAi
    else:

        variance = sum(dot(PHI,SIGMAi) * PHI, 1).reshape(n, 1)
        sigma = variance+exp(-lnBeta)

        trainRMSE = sqrt(mean(omega[training] * delta ** 2))
        trainLL = mean(-0.5 * delta ** 2/sigma - 0.5 * log(sigma))-0.5*log(2*pi)

        if len(validation)>0:
            n = sum(validation)

            phi = exp(-0.5 * Dxy(X[validation, :], P) * gamma ** 2)

            lnBeta = b+log(omega[validation])

            if(heteroscedastic):
                lnBeta = dot(phi, u)+lnBeta

            if (joint):
                phi = hstack([phi, X[validation, :], ones((n, 1))])

            variance = sum(dot(phi,SIGMAi) * phi, 1).reshape(n, 1)
            sigma = variance+exp(-lnBeta)

            delta = dot(phi, w) - y[validation, :]

            validRMSE = sqrt(mean(omega[validation] * delta ** 2))
            validLL = mean(-0.5 * delta ** 2/sigma - 0.5 * log(sigma))-0.5*log(2*pi)

        return nlogML


def GL_grad(theta, m, X, y, omega=[], joint=False,heteroscedastic=False, training=[], validation=[]):

    n, d = X.shape

    if training==[]:
        training = ones(n, dtype=bool)

    if omega==[]:
        omega = ones((n, 1))

    n = sum(training)
    d = len(X[0, :])

    if(joint):
        a_dim = m+d+1
    else:
        a_dim = m

    P = theta[0:m * d].reshape(m, d)
    gamma = theta[m*d]
    lnAlpha = theta[m*d+1:m*d+1+a_dim].reshape(a_dim, 1)
    b = theta[m*d+1+a_dim]

    alpha = exp(lnAlpha)

    D = Dxy(X[training, :], P)

    lnPHI = -0.5 * D * gamma ** 2
    PHI = exp(lnPHI)

    lnBeta = b+log(omega[training])

    if(heteroscedastic):
        u = theta[m*d+1+a_dim+1:m*d+1+a_dim+1+m].reshape(m, 1)
        lnBeta = lnBeta+dot(PHI, u)

    beta = exp(lnBeta)

    if (joint):
        PHI = hstack([PHI, X[training, :], ones((n, 1))])

    A = diag(alpha[:, 0])

    PSi = PHI * beta
    PSiY = dot(PSi.T, y[training, :])

    SIGMA = dot(PSi.T, PHI) + A
    Li = inv(cholesky(SIGMA))

    SIGMAi = dot(Li.T,Li)

    w = dot(SIGMAi, PSiY)

    delta = dot(PHI, w) - y[training, :]

    dwda = -dot(SIGMAi, alpha * w)

    dlnAlpha = -sum(dot(dwda, (beta * delta).T) * PHI.T, 1).reshape(len(alpha),1) - alpha * w * dwda - 0.5*alpha * w ** 2 -0.5*diag(SIGMAi).reshape(len(alpha), 1) * alpha + 0.5

    PHIxSIGMAi = dot(PHI, SIGMAi)

    variance = sum(PHIxSIGMAi * PHI, 1).reshape(n, 1)
    dbeta = -0.5 * (variance + delta ** 2) * beta + 0.5
    db = sum(dbeta)

    if(heteroscedastic):
        lnEta = theta[m*d+1+a_dim+1+m:m*d+1+a_dim+1+m+m].reshape(m, 1)
        eta = exp(lnEta)
        du = dot(PHI[:, 0:m].T, dbeta)-u*eta
        dlnEta = -0.5*eta*u**2+0.5
        dlnPHI = (dot(dbeta, u.T) - dot(delta * beta, w[0:m].T) - dot(PSi, SIGMAi[:, 0:m])) * PHI[:, 0:m]
    else:
        dlnPHI = (- dot(delta * beta, w[0:m].T) - dot(PSi, SIGMAi[:, 0:m])) * PHI[:, 0:m]

    dP = zeros((m, d))

    for j in range(m):
        Delta = X[training, :] - P[j, :]
        dP[j, :] = sum(Delta * dlnPHI[:, j].reshape(n, 1), 0) * gamma ** 2

    dg = -gamma * sum(dlnPHI * D)

    if(heteroscedastic):
        grad = concatenate((dP.flatten(), dg.flatten(), dlnAlpha.flatten(), db.flatten(), du.flatten(),dlnEta.flatten()))
    else:
        grad = concatenate((dP.flatten(), dg.flatten(), dlnAlpha.flatten(), db.flatten()))

    return -grad/n

def VL(theta, m, X, y, omega=[], joint=False,heteroscedastic=False, training=[], validation=[], returnModel=False):

    global nlogML

    global trainRMSE
    global trainLL

    global validRMSE
    global validLL

    n,d = X.shape
    if training==[]:
        training = ones(n, dtype=bool)

    if omega==[]:
        omega = ones((n, 1))

    n = sum(training)

    if(joint):
        a_dim = m+d+1
    else:
        a_dim = m

    P = theta[0:m * d].reshape(m, d)
    gamma = theta[m*d:m*d+m]
    lnAlpha = theta[m*d+m:m*d+m+a_dim].reshape(a_dim, 1)
    b = theta[m*d+m+a_dim]

    alpha = exp(lnAlpha)

    lnPHI = -0.5 * Dxy(X[training, :], P) * gamma ** 2
    PHI = exp(lnPHI)

    lnBeta = b+log(omega[training])

    if(heteroscedastic):
        u = theta[m*d+m+a_dim+1:m*d+m+a_dim+m+1].reshape(m, 1)
        lnBeta = dot(PHI, u)+lnBeta

    beta = exp(lnBeta)

    if (joint):
        PHI = hstack([PHI, X[training, :], ones((n, 1))])


    if y==[]:
        return PHI,lnBeta

    A = diag(alpha[:, 0])

    PSi = PHI * (beta)
    PSiY = dot(PSi.T, y[training, :])

    SIGMA = dot(PSi.T, PHI) + A

    Li = inv(cholesky(SIGMA))

    SIGMAi = dot(Li.T,Li)

    logdet = -2*sum(log(diag(Li)))

    w = dot(SIGMAi, PSiY)

    pred = dot(PHI, w)
    delta = pred - y[training, :]

    nlogML = -0.5 * sum(delta ** 2 * beta) + 0.5 * sum(lnBeta) - 0.5 * sum(w ** 2 * alpha) + 0.5 * sum(lnAlpha) - 0.5 * logdet - (n / 2) * log(2 * pi)

    if(heteroscedastic):
        lnEta = theta[m*d+m+a_dim+m+1:m*d+m+a_dim+m+m+1].reshape(m, 1)
        eta = exp(lnEta)
        nlogML = nlogML - 0.5 * sum(u ** 2 * eta) + 0.5 * sum(lnEta)

    nlogML = -nlogML/n

    if returnModel:
        return w, SIGMAi
    else:

        variance = sum(dot(PHI,SIGMAi) * PHI, 1).reshape(n, 1)
        sigma = variance+exp(-lnBeta)

        trainRMSE = sqrt(mean(omega[training] * delta ** 2))
        trainLL = mean(-0.5 * delta ** 2/sigma - 0.5 * log(sigma))-0.5*log(2*pi)

        if len(validation)>0:
            n = sum(validation)

            phi = exp(-0.5 * Dxy(X[validation, :], P) * gamma ** 2)

            lnBeta = b+log(omega[validation])

            if(heteroscedastic):
                lnBeta = dot(phi, u)+lnBeta

            if (joint):
                phi = hstack([phi, X[validation, :], ones((n, 1))])

            variance = sum(dot(phi,SIGMAi) * phi, 1).reshape(n, 1)
            sigma = variance+exp(-lnBeta)

            delta = dot(phi, w) - y[validation, :]

            validRMSE = sqrt(mean(omega[validation] * delta ** 2))
            validLL = mean(-0.5 * delta ** 2/sigma - 0.5 * log(sigma))-0.5*log(2*pi)

        return nlogML


def VL_grad(theta, m, X, y, omega=[], joint=False,heteroscedastic=False, training=[], validation=[]):

    n, d = X.shape

    if training==[]:
        training = ones(n, dtype=bool)

    if omega==[]:
        omega = ones((n, 1))

    n = sum(training)
    d = len(X[0, :])

    if(joint):
        a_dim = m+d+1
    else:
        a_dim = m

    P = theta[0:m * d].reshape(m, d)
    gamma = theta[m*d:m*d+m]
    lnAlpha = theta[m*d+m:m*d+m+a_dim].reshape(a_dim, 1)
    b = theta[m*d+m+a_dim]

    alpha = exp(lnAlpha)

    D = Dxy(X[training, :], P)

    lnPHI = -0.5 * D * gamma ** 2
    PHI = exp(lnPHI)

    lnBeta = b+log(omega[training])

    if(heteroscedastic):
        u = theta[m*d+m+a_dim+1:m*d+m+a_dim+m+1].reshape(m, 1)
        lnBeta = lnBeta+dot(PHI, u)

    beta = exp(lnBeta)

    if (joint):
        PHI = hstack([PHI, X[training, :], ones((n, 1))])

    A = diag(alpha[:, 0])

    PSi = PHI * beta
    PSiY = dot(PSi.T, y[training, :])

    SIGMA = dot(PSi.T, PHI) + A
    Li = inv(cholesky(SIGMA))

    SIGMAi = dot(Li.T,Li)

    w = dot(SIGMAi, PSiY)

    delta = dot(PHI, w) - y[training, :]

    dwda = -dot(SIGMAi, alpha * w)

    dlnAlpha = -sum(dot(dwda, (beta * delta).T) * PHI.T, 1).reshape(len(alpha),1) - alpha * w * dwda - 0.5*alpha * w ** 2 -0.5*diag(
        SIGMAi).reshape(len(alpha), 1) * alpha + 0.5

    PHIxSIGMAi = dot(PHI, SIGMAi)

    variance = sum(PHIxSIGMAi * PHI, 1).reshape(n, 1)
    dbeta = -0.5 * (variance + delta ** 2) * beta + 0.5
    db = sum(dbeta)

    if(heteroscedastic):
        lnEta = theta[m*d+m+a_dim+m+1:m*d+m+a_dim+m+m+1].reshape(m, 1)
        eta = exp(lnEta)
        du = dot(PHI[:, 0:m].T, dbeta)-u*eta
        dlnEta = -0.5*eta*u**2+0.5
        dlnPHI = (dot(dbeta, u.T) - dot(delta * beta, w[0:m].T) - dot(PSi, SIGMAi[:, 0:m])) * PHI[:, 0:m]
    else:
        dlnPHI = (- dot(delta * beta, w[0:m].T) - dot(PSi, SIGMAi[:, 0:m])) * PHI[:, 0:m]

    dP = zeros((m, d))

    for j in range(m):
        Delta = X[training, :] - P[j, :]
        dP[j, :] = sum(Delta * dlnPHI[:, j].reshape(n, 1), 0) * gamma[j] ** 2

    dg = -gamma * sum(dlnPHI * D,0)

    if(heteroscedastic):
        grad = concatenate((dP.flatten(), dg.flatten(), dlnAlpha.flatten(), db.flatten(), du.flatten(),dlnEta.flatten()))
    else:
        grad = concatenate((dP.flatten(), dg.flatten(), dlnAlpha.flatten(), db.flatten()))

    return -grad/n

def GC(theta, m, X, y, omega=[], joint=False,heteroscedastic=False, training=[], validation=[], returnModel=False):

    global nlogML

    global trainRMSE
    global trainLL

    global validRMSE
    global validLL

    n,d = X.shape
    if training==[]:
        training = ones(n, dtype=bool)

    if omega==[]:
        omega = ones((n, 1))

    n = sum(training)

    if(joint):
        a_dim = m+d+1
    else:
        a_dim = m

    P = theta[0:m * d].reshape(m, d)
    GAMMA = theta[m*d:m*d+d*d].reshape(d, d)
    lnAlpha = theta[m*d+d*d:m*d+d*d+a_dim].reshape(a_dim, 1)
    b = theta[m*d+d*d+a_dim]

    alpha = exp(lnAlpha)

    PHI = zeros((n, m))

    for j in range(m):
        Delta = X[training, :] - P[j, :]
        PHI[:, j] = exp(-0.5 * sum(power(dot(Delta, GAMMA.T), 2), 1))

    lnBeta = b+log(omega[training])

    if(heteroscedastic):
        u = theta[m*d+d*d+a_dim+1:m*d+d*d+a_dim+m+1].reshape(m, 1)
        lnBeta = dot(PHI, u)+lnBeta

    beta = exp(lnBeta)

    if (joint):
        PHI = hstack([PHI, X[training, :], ones((n, 1))])


    if y==[]:
        return PHI,lnBeta

    A = diag(alpha[:, 0])

    PSi = PHI * (beta)
    PSiY = dot(PSi.T, y[training, :])

    SIGMA = dot(PSi.T, PHI) + A

    Li = inv(cholesky(SIGMA))

    SIGMAi = dot(Li.T,Li)

    logdet = -2*sum(log(diag(Li)))

    w = dot(SIGMAi, PSiY)

    pred = dot(PHI, w)
    delta = pred - y[training, :]

    nlogML = -0.5 * sum(delta ** 2 * beta) + 0.5 * sum(lnBeta) - 0.5 * sum(w ** 2 * alpha) + 0.5 * sum(lnAlpha) - 0.5 * logdet - (n / 2) * log(2 * pi)

    if(heteroscedastic):
        lnEta = theta[m*d+d*d+a_dim+m+1:m*d+d*d+a_dim+m+m+1].reshape(m, 1)
        eta = exp(lnEta)
        nlogML = nlogML - 0.5 * sum(u ** 2 * eta) + 0.5 * sum(lnEta)


    nlogML = -nlogML/n

    if returnModel:
        return w, SIGMAi
    else:

        variance = sum(dot(PHI,SIGMAi) * PHI, 1).reshape(n, 1)
        sigma = variance+exp(-lnBeta)

        trainRMSE = sqrt(mean(omega[training] * delta ** 2))
        trainLL = mean(-0.5 * delta ** 2/sigma - 0.5 * log(sigma))-0.5*log(2*pi)

        if len(validation)>0:
            n = sum(validation)

            phi = zeros((n, m))

            for j in range(m):
                Delta = X[validation, :] - P[j, :]
                phi[:, j] = exp(-0.5 * sum(power(dot(Delta, GAMMA.T), 2), 1))

            lnBeta = b+log(omega[validation])

            if(heteroscedastic):
                lnBeta = dot(phi, u)+lnBeta

            if (joint):
                phi = hstack([phi, X[validation, :], ones((n, 1))])

            variance = sum(dot(phi,SIGMAi) * phi, 1).reshape(n, 1)
            sigma = variance+exp(-lnBeta)

            delta = dot(phi, w) - y[validation, :]

            validRMSE = sqrt(mean(omega[validation] * delta ** 2))
            validLL = mean(-0.5 * delta ** 2/sigma - 0.5 * log(sigma))-0.5*log(2*pi)

        return nlogML


def GC_grad(theta, m, X, y, omega=[], joint=False,heteroscedastic=False, training=[], validation=[]):

    n, d = X.shape

    if training==[]:
        training = ones(n, dtype=bool)

    if omega==[]:
        omega = ones((n, 1))

    n = sum(training)
    d = len(X[0, :])

    if(joint):
        a_dim = m+d+1
    else:
        a_dim = m

    P = theta[0:m * d].reshape(m, d)
    GAMMA = theta[m*d:m*d+d*d].reshape(d, d)
    lnAlpha = theta[m*d+d*d:m*d+d*d+a_dim].reshape(a_dim, 1)
    b = theta[m*d+d*d+a_dim]

    alpha = exp(lnAlpha)

    PHI = zeros((n, m))

    for j in range(m):
        Delta = X[training, :] - P[j, :]
        PHI[:, j] = exp(-0.5 * sum(power(dot(Delta, GAMMA.T), 2), 1))

    lnBeta = b+log(omega[training])

    if(heteroscedastic):
        u = theta[m*d+d*d+a_dim+1:m*d+d*d+a_dim+m+1].reshape(m, 1)
        lnBeta = lnBeta+dot(PHI, u)

    beta = exp(lnBeta)

    if (joint):
        PHI = hstack([PHI, X[training, :], ones((n, 1))])

    A = diag(alpha[:, 0])

    PSi = PHI * beta
    PSiY = dot(PSi.T, y[training, :])

    SIGMA = dot(PSi.T, PHI) + A
    Li = inv(cholesky(SIGMA))

    SIGMAi = dot(Li.T,Li)

    w = dot(SIGMAi, PSiY)

    delta = dot(PHI, w) - y[training, :]

    dwda = -dot(SIGMAi, alpha * w)

    dlnAlpha = -sum(dot(dwda, (beta * delta).T) * PHI.T, 1).reshape(len(alpha),1) - alpha * w * dwda - 0.5*alpha * w ** 2 -0.5*diag(
        SIGMAi).reshape(len(alpha), 1) * alpha + 0.5

    PHIxSIGMAi = dot(PHI, SIGMAi)

    variance = sum(PHIxSIGMAi * PHI, 1).reshape(n, 1)
    dbeta = -0.5 * (variance + delta ** 2) * beta + 0.5
    db = sum(dbeta)

    if(heteroscedastic):
        lnEta = theta[m*d+d*d+a_dim+m+1:m*d+d*d+a_dim+m+m+1].reshape(m, 1)
        eta = exp(lnEta)
        du = dot(PHI[:, 0:m].T, dbeta)-u*eta
        dlnEta = -0.5*eta*u**2+0.5
        dlnPHI = (dot(dbeta, u.T) - dot(delta * beta, w[0:m].T) - dot(PSi, SIGMAi[:, 0:m])) * PHI[:, 0:m]
    else:
        dlnPHI = (- dot(delta * beta, w[0:m].T) - dot(PSi, SIGMAi[:, 0:m])) * PHI[:, 0:m]

    dP = zeros((m, d))
    G = dot(GAMMA.T,GAMMA)
    dG = zeros((d, d))

    for j in range(m):
        Delta = X[training, :] - P[j, :]
        dP[j, :] = dot(dot(dlnPHI[:,j].T,Delta),G)
        dG = dG-dot(dot(GAMMA,(Delta*dlnPHI[:,j].reshape(n, 1)).T),Delta)

    if(heteroscedastic):
        grad = concatenate((dP.flatten(), dG.flatten(), dlnAlpha.flatten(), db.flatten(), du.flatten(),dlnEta.flatten()))
    else:
        grad = concatenate((dP.flatten(), dG.flatten(), dlnAlpha.flatten(), db.flatten()))

    return -grad/n

def VC(theta, m, X, y, omega=[], joint=False,heteroscedastic=False, training=[], validation=[], returnModel=False):

    global nlogML

    global trainRMSE
    global trainLL

    global validRMSE
    global validLL

    n,d = X.shape
    if training==[]:
        training = ones(n, dtype=bool)

    if omega==[]:
        omega = ones((n, 1))

    n = sum(training)

    if(joint):
        a_dim = m+d+1
    else:
        a_dim = m

    P = theta[0:m * d].reshape(m, d)
    GAMMA = theta[m*d:m*d+d*d*m].reshape(d, d, m)
    lnAlpha = theta[m*d+d*d*m:m*d+d*d*m+a_dim].reshape(a_dim, 1)
    b = theta[m*d+d*d*m+a_dim]

    alpha = exp(lnAlpha)

    PHI = zeros((n, m))

    for j in range(m):
        Delta = X[training, :] - P[j, :]
        PHI[:, j] = exp(-0.5 * sum(power(dot(Delta, GAMMA[:, :, j].T), 2), 1))

    lnBeta = b+log(omega[training])

    if(heteroscedastic):
        u = theta[m*d+d*d*m+a_dim+1:m*d+d*d*m+a_dim+m+1].reshape(m, 1)
        lnBeta = dot(PHI, u)+lnBeta

    beta = exp(lnBeta)

    if (joint):
        PHI = hstack([PHI, X[training, :], ones((n, 1))])


    if y==[]:
        return PHI,lnBeta

    A = diag(alpha[:, 0])

    PSi = PHI * (beta)
    PSiY = dot(PSi.T, y[training, :])

    SIGMA = dot(PSi.T, PHI) + A

    Li = inv(cholesky(SIGMA))

    SIGMAi = dot(Li.T,Li)

    logdet = -2*sum(log(diag(Li)))

    w = dot(SIGMAi, PSiY)

    pred = dot(PHI, w)
    delta = pred - y[training, :]

    nlogML = -0.5 * sum(delta ** 2 * beta) + 0.5 * sum(lnBeta) - 0.5 * sum(w ** 2 * alpha) + 0.5 * sum(lnAlpha) - 0.5 * logdet - (n / 2) * log(2 * pi)

    if(heteroscedastic):
        lnEta = theta[m*d+d*d*m+a_dim+m+1:m*d+d*d*m+a_dim+m+m+1].reshape(m, 1)
        eta = exp(lnEta)
        nlogML = nlogML - 0.5 * sum(u ** 2 * eta) + 0.5 * sum(lnEta)


    nlogML = -nlogML/n

    if returnModel:
        return w, SIGMAi
    else:

        variance = sum(dot(PHI,SIGMAi) * PHI, 1).reshape(n, 1)
        sigma = variance+exp(-lnBeta)

        trainRMSE = sqrt(mean(omega[training] * delta ** 2))
        trainLL = mean(-0.5 * delta ** 2/sigma - 0.5 * log(sigma))-0.5*log(2*pi)

        if len(validation)>0:
            n = sum(validation)

            phi = zeros((n, m))

            for j in range(m):
                Delta = X[validation, :] - P[j, :]
                phi[:, j] = exp(-0.5 * sum(power(dot(Delta, GAMMA[:, :, j].T), 2), 1))

            lnBeta = b+log(omega[validation])

            if(heteroscedastic):
                lnBeta = dot(phi, u)+lnBeta

            if (joint):
                phi = hstack([phi, X[validation, :], ones((n, 1))])

            variance = sum(dot(phi,SIGMAi) * phi, 1).reshape(n, 1)
            sigma = variance+exp(-lnBeta)

            delta = dot(phi, w) - y[validation, :]

            validRMSE = sqrt(mean(omega[validation] * delta ** 2))
            validLL = mean(-0.5 * delta ** 2/sigma - 0.5 * log(sigma))-0.5*log(2*pi)

        return nlogML


def VC_grad(theta, m, X, y, omega=[], joint=False,heteroscedastic=False, training=[], validation=[]):

    n, d = X.shape

    if training==[]:
        training = ones(n, dtype=bool)

    if omega==[]:
        omega = ones((n, 1))

    n = sum(training)
    d = len(X[0, :])

    if(joint):
        a_dim = m+d+1
    else:
        a_dim = m

    P = theta[0:m * d].reshape(m, d)
    GAMMA = theta[m*d:m*d+d*d*m].reshape(d, d, m)
    lnAlpha = theta[m*d+d*d*m:m*d+d*d*m+a_dim].reshape(a_dim, 1)
    b = theta[m*d+d*d*m+a_dim]

    alpha = exp(lnAlpha)

    PHI = zeros((n, m))

    for j in range(m):
        Delta = X[training, :] - P[j, :]
        PHI[:, j] = exp(-0.5 * sum(power(dot(Delta, GAMMA[:, :, j].T), 2), 1))

    lnBeta = b+log(omega[training])

    if(heteroscedastic):
        u = theta[m*d+d*d*m+a_dim+1:m*d+d*d*m+a_dim+m+1].reshape(m, 1)
        lnBeta = lnBeta+dot(PHI, u)

    beta = exp(lnBeta)

    if (joint):
        PHI = hstack([PHI, X[training, :], ones((n, 1))])

    A = diag(alpha[:, 0])

    PSi = PHI * beta
    PSiY = dot(PSi.T, y[training, :])

    SIGMA = dot(PSi.T, PHI) + A
    Li = inv(cholesky(SIGMA))

    SIGMAi = dot(Li.T,Li)

    w = dot(SIGMAi, PSiY)

    delta = dot(PHI, w) - y[training, :]

    dwda = -dot(SIGMAi, alpha * w)

    dlnAlpha = -sum(dot(dwda, (beta * delta).T) * PHI.T, 1).reshape(len(alpha),1) - alpha * w * dwda - 0.5*alpha * w ** 2 -0.5*diag(
        SIGMAi).reshape(len(alpha), 1) * alpha + 0.5

    PHIxSIGMAi = dot(PHI, SIGMAi)

    variance = sum(PHIxSIGMAi * PHI, 1).reshape(n, 1)
    dbeta = -0.5 * (variance + delta ** 2) * beta + 0.5
    db = sum(dbeta)

    if(heteroscedastic):
        lnEta = theta[m*d+d*d*m+a_dim+m+1:m*d+d*d*m+a_dim+m+m+1].reshape(m, 1)
        eta = exp(lnEta)
        du = dot(PHI[:, 0:m].T, dbeta)-u*eta
        dlnEta = -0.5*eta*u**2+0.5
        dlnPHI = (dot(dbeta, u.T) - dot(delta * beta, w[0:m].T) - dot(PSi, SIGMAi[:, 0:m])) * PHI[:, 0:m]
    else:
        dlnPHI = (- dot(delta * beta, w[0:m].T) - dot(PSi, SIGMAi[:, 0:m])) * PHI[:, 0:m]

    dP = zeros((m, d))
    dG = zeros((d, d, m))

    for j in range(m):
        Delta = X[training, :] - P[j, :]
        dP[j, :] = dot(dot(dlnPHI[:,j].T,Delta),dot(GAMMA[:, :, j].T,GAMMA[:, :, j]))
        dG[:, :, j] = -dot(dot(GAMMA[:, :, j],(Delta*dlnPHI[:,j].reshape(n, 1)).T),Delta)

    if(heteroscedastic):
        grad = concatenate((dP.flatten(), dG.flatten(), dlnAlpha.flatten(), db.flatten(), du.flatten(),dlnEta.flatten()))
    else:
        grad = concatenate((dP.flatten(), dG.flatten(), dlnAlpha.flatten(), db.flatten()))

    return -grad/n

def GD(theta, m, X, y, omega=[], joint=False,heteroscedastic=False, training=[], validation=[], returnModel=False):

    global nlogML

    global trainRMSE
    global trainLL

    global validRMSE
    global validLL

    n,d = X.shape
    if training==[]:
        training = ones(n, dtype=bool)

    if omega==[]:
        omega = ones((n, 1))

    n = sum(training)

    if(joint):
        a_dim = m+d+1
    else:
        a_dim = m

    P = theta[0:m * d].reshape(m, d)
    GAMMA = theta[m*d:m*d+d].reshape(1, d)
    lnAlpha = theta[m*d+d:m*d+d+a_dim].reshape(a_dim, 1)
    b = theta[m*d+d+a_dim]

    alpha = exp(lnAlpha)

    PHI = zeros((n, m))

    for j in range(m):
        Delta = X[training, :] - P[j, :]
        PHI[:, j] = exp(-0.5 * sum(power(Delta*GAMMA, 2), 1))

    lnBeta = b+log(omega[training])

    if(heteroscedastic):
        u = theta[m*d+d+a_dim+1:m*d+d+a_dim+m+1].reshape(m, 1)
        lnBeta = dot(PHI, u)+lnBeta

    beta = exp(lnBeta)

    if (joint):
        PHI = hstack([PHI, X[training, :], ones((n, 1))])


    if y==[]:
        return PHI,lnBeta

    A = diag(alpha[:, 0])

    PSi = PHI * (beta)
    PSiY = dot(PSi.T, y[training, :])

    SIGMA = dot(PSi.T, PHI) + A

    Li = inv(cholesky(SIGMA))

    SIGMAi = dot(Li.T,Li)

    logdet = -2*sum(log(diag(Li)))

    w = dot(SIGMAi, PSiY)

    pred = dot(PHI, w)
    delta = pred - y[training, :]

    nlogML = -0.5 * sum(delta ** 2 * beta) + 0.5 * sum(lnBeta) - 0.5 * sum(w ** 2 * alpha) + 0.5 * sum(lnAlpha) - 0.5 * logdet - (n / 2) * log(2 * pi)

    if(heteroscedastic):
        lnEta = theta[m*d+d+a_dim+m+1:m*d+d+a_dim+m+m+1].reshape(m, 1)
        eta = exp(lnEta)
        nlogML = nlogML - 0.5 * sum(u ** 2 * eta) + 0.5 * sum(lnEta)


    nlogML = -nlogML/n

    if returnModel:
        return w, SIGMAi
    else:

        variance = sum(dot(PHI,SIGMAi) * PHI, 1).reshape(n, 1)
        sigma = variance+exp(-lnBeta)

        trainRMSE = sqrt(mean(omega[training] * delta ** 2))
        trainLL = mean(-0.5 * delta ** 2/sigma - 0.5 * log(sigma))-0.5*log(2*pi)

        if len(validation)>0:
            n = sum(validation)

            phi = zeros((n, m))

            for j in range(m):
                Delta = X[validation, :] - P[j, :]
                phi[:, j] = exp(-0.5 * sum(power(Delta*GAMMA, 2), 1))

            lnBeta = b+log(omega[validation])

            if(heteroscedastic):
                lnBeta = dot(phi, u)+lnBeta

            if (joint):
                phi = hstack([phi, X[validation, :], ones((n, 1))])

            variance = sum(dot(phi,SIGMAi) * phi, 1).reshape(n, 1)
            sigma = variance+exp(-lnBeta)

            delta = dot(phi, w) - y[validation, :]

            validRMSE = sqrt(mean(omega[validation] * delta ** 2))
            validLL = mean(-0.5 * delta ** 2/sigma - 0.5 * log(sigma))-0.5*log(2*pi)

        return nlogML


def GD_grad(theta, m, X, y, omega=[], joint=False,heteroscedastic=False, training=[], validation=[]):

    n, d = X.shape

    if training==[]:
        training = ones(n, dtype=bool)

    if omega==[]:
        omega = ones((n, 1))

    n = sum(training)
    d = len(X[0, :])

    if(joint):
        a_dim = m+d+1
    else:
        a_dim = m

    P = theta[0:m * d].reshape(m, d)
    GAMMA = theta[m*d:m*d+d].reshape(1, d)
    lnAlpha = theta[m*d+d:m*d+d+a_dim].reshape(a_dim, 1)
    b = theta[m*d+d+a_dim]

    alpha = exp(lnAlpha)

    PHI = zeros((n, m))

    for j in range(m):
        Delta = X[training, :] - P[j, :]
        PHI[:, j] = exp(-0.5 * sum(power(Delta*GAMMA, 2), 1))

    lnBeta = b+log(omega[training])

    if(heteroscedastic):
        u = theta[m*d+d+a_dim+1:m*d+d+a_dim+m+1].reshape(m, 1)
        lnBeta = lnBeta+dot(PHI, u)

    beta = exp(lnBeta)

    if (joint):
        PHI = hstack([PHI, X[training, :], ones((n, 1))])

    A = diag(alpha[:, 0])

    PSi = PHI * beta
    PSiY = dot(PSi.T, y[training, :])

    SIGMA = dot(PSi.T, PHI) + A
    Li = inv(cholesky(SIGMA))

    SIGMAi = dot(Li.T,Li)

    w = dot(SIGMAi, PSiY)

    delta = dot(PHI, w) - y[training, :]

    dwda = -dot(SIGMAi, alpha * w)

    dlnAlpha = -sum(dot(dwda, (beta * delta).T) * PHI.T, 1).reshape(len(alpha),1) - alpha * w * dwda - 0.5*alpha * w ** 2 -0.5*diag(
        SIGMAi).reshape(len(alpha), 1) * alpha + 0.5

    PHIxSIGMAi = dot(PHI, SIGMAi)

    variance = sum(PHIxSIGMAi * PHI, 1).reshape(n, 1)
    dbeta = -0.5 * (variance + delta ** 2) * beta + 0.5
    db = sum(dbeta)

    if(heteroscedastic):
        lnEta = theta[m*d+d+a_dim+m+1:m*d+d+a_dim+m+m+1].reshape(m, 1)
        eta = exp(lnEta)
        du = dot(PHI[:, 0:m].T, dbeta)-u*eta
        dlnEta = -0.5*eta*u**2+0.5
        dlnPHI = (dot(dbeta, u.T) - dot(delta * beta, w[0:m].T) - dot(PSi, SIGMAi[:, 0:m])) * PHI[:, 0:m]
    else:
        dlnPHI = (- dot(delta * beta, w[0:m].T) - dot(PSi, SIGMAi[:, 0:m])) * PHI[:, 0:m]

    dP = zeros((m, d))
    dG = zeros((1,d))

    for j in range(m):
        Delta = X[training, :] - P[j, :]
        dP[j, :] = dot(dlnPHI[:,j].T,Delta)*GAMMA**2
        dG = dG-diag(dot(GAMMA.reshape(d,1)*(Delta*dlnPHI[:,j].reshape(n, 1)).T,Delta))

    if(heteroscedastic):
        grad = concatenate((dP.flatten(), dG.flatten(), dlnAlpha.flatten(), db.flatten(), du.flatten(),dlnEta.flatten()))
    else:
        grad = concatenate((dP.flatten(), dG.flatten(), dlnAlpha.flatten(), db.flatten()))

    return -grad/n

def VD(theta, m, X, y, omega=[], joint=False,heteroscedastic=False, training=[], validation=[], returnModel=False):

    global nlogML

    global trainRMSE
    global trainLL

    global validRMSE
    global validLL

    n,d = X.shape
    if training==[]:
        training = ones(n, dtype=bool)

    if omega==[]:
        omega = ones((n, 1))

    n = sum(training)

    if(joint):
        a_dim = m+d+1
    else:
        a_dim = m

    P = theta[0:m * d].reshape(m, d)
    GAMMA = theta[m*d:m*d+m*d].reshape(m, d)
    lnAlpha = theta[m*d+m*d:m*d+m*d+a_dim].reshape(a_dim, 1)
    b = theta[m*d+m*d+a_dim]

    alpha = exp(lnAlpha)

    PHI = zeros((n, m))

    for j in range(m):
        Delta = X[training, :] - P[j, :]
        PHI[:, j] = exp(-0.5 * sum(power(Delta*GAMMA[j,:], 2), 1))

    lnBeta = b+log(omega[training])

    if(heteroscedastic):
        u = theta[m*d+m*d+a_dim+1:m*d+m*d+a_dim+m+1].reshape(m, 1)
        lnBeta = dot(PHI, u)+lnBeta

    beta = exp(lnBeta)

    if (joint):
        PHI = hstack([PHI, X[training, :], ones((n, 1))])


    if y==[]:
        return PHI,lnBeta

    A = diag(alpha[:, 0])

    PSi = PHI * (beta)
    PSiY = dot(PSi.T, y[training, :])

    SIGMA = dot(PSi.T, PHI) + A

    Li = inv(cholesky(SIGMA))

    SIGMAi = dot(Li.T,Li)

    logdet = -2*sum(log(diag(Li)))

    w = dot(SIGMAi, PSiY)

    pred = dot(PHI, w)
    delta = pred - y[training, :]

    nlogML = -0.5 * sum(delta ** 2 * beta) + 0.5 * sum(lnBeta) - 0.5 * sum(w ** 2 * alpha) + 0.5 * sum(lnAlpha) - 0.5 * logdet - (n / 2) * log(2 * pi)

    if(heteroscedastic):
        lnEta = theta[m*d+m*d+a_dim+m+1:m*d+m*d+a_dim+m+m+1].reshape(m, 1)
        eta = exp(lnEta)
        nlogML = nlogML - 0.5 * sum(u ** 2 * eta) + 0.5 * sum(lnEta)


    nlogML = -nlogML/n

    if returnModel:
        return w, SIGMAi
    else:

        variance = sum(dot(PHI,SIGMAi) * PHI, 1).reshape(n, 1)
        sigma = variance+exp(-lnBeta)

        trainRMSE = sqrt(mean(omega[training] * delta ** 2))
        trainLL = mean(-0.5 * delta ** 2/sigma - 0.5 * log(sigma))-0.5*log(2*pi)

        if len(validation)>0:
            n = sum(validation)

            phi = zeros((n, m))

            for j in range(m):
                Delta = X[validation, :] - P[j, :]
                phi[:, j] = exp(-0.5 * sum(power(Delta*GAMMA[j,:], 2), 1))

            lnBeta = b+log(omega[validation])

            if(heteroscedastic):
                lnBeta = dot(phi, u)+lnBeta

            if (joint):
                phi = hstack([phi, X[validation, :], ones((n, 1))])

            variance = sum(dot(phi,SIGMAi) * phi, 1).reshape(n, 1)
            sigma = variance+exp(-lnBeta)

            delta = dot(phi, w) - y[validation, :]

            validRMSE = sqrt(mean(omega[validation] * delta ** 2))
            validLL = mean(-0.5 * delta ** 2/sigma - 0.5 * log(sigma))-0.5*log(2*pi)

        return nlogML


def VD_grad(theta, m, X, y, omega=[], joint=False,heteroscedastic=False, training=[], validation=[]):

    n, d = X.shape

    if training==[]:
        training = ones(n, dtype=bool)

    if omega==[]:
        omega = ones((n, 1))

    n = sum(training)
    d = len(X[0, :])

    if(joint):
        a_dim = m+d+1
    else:
        a_dim = m

    P = theta[0:m * d].reshape(m, d)
    GAMMA = theta[m*d:m*d+m*d].reshape(m, d)
    lnAlpha = theta[m*d+m*d:m*d+m*d+a_dim].reshape(a_dim, 1)
    b = theta[m*d+m*d+a_dim]

    alpha = exp(lnAlpha)

    PHI = zeros((n, m))

    for j in range(m):
        Delta = X[training, :] - P[j, :]
        PHI[:, j] = exp(-0.5 * sum(power(Delta*GAMMA[j,:], 2), 1))

    lnBeta = b+log(omega[training])

    if(heteroscedastic):
        u = theta[m*d+m*d+a_dim+1:m*d+m*d+a_dim+m+1].reshape(m, 1)
        lnBeta = lnBeta+dot(PHI, u)

    beta = exp(lnBeta)

    if (joint):
        PHI = hstack([PHI, X[training, :], ones((n, 1))])

    A = diag(alpha[:, 0])

    PSi = PHI * beta
    PSiY = dot(PSi.T, y[training, :])

    SIGMA = dot(PSi.T, PHI) + A
    Li = inv(cholesky(SIGMA))

    SIGMAi = dot(Li.T,Li)

    w = dot(SIGMAi, PSiY)

    delta = dot(PHI, w) - y[training, :]

    dwda = -dot(SIGMAi, alpha * w)

    dlnAlpha = -sum(dot(dwda, (beta * delta).T) * PHI.T, 1).reshape(len(alpha),1) - alpha * w * dwda - 0.5*alpha * w ** 2 -0.5*diag(
        SIGMAi).reshape(len(alpha), 1) * alpha + 0.5

    PHIxSIGMAi = dot(PHI, SIGMAi)

    variance = sum(PHIxSIGMAi * PHI, 1).reshape(n, 1)
    dbeta = -0.5 * (variance + delta ** 2) * beta + 0.5
    db = sum(dbeta)

    if(heteroscedastic):
        lnEta = theta[m*d+m*d+a_dim+m+1:m*d+m*d+a_dim+m+m+1].reshape(m, 1)
        eta = exp(lnEta)
        du = dot(PHI[:, 0:m].T, dbeta)-u*eta
        dlnEta = -0.5*eta*u**2+0.5
        dlnPHI = (dot(dbeta, u.T) - dot(delta * beta, w[0:m].T) - dot(PSi, SIGMAi[:, 0:m])) * PHI[:, 0:m]
    else:
        dlnPHI = (- dot(delta * beta, w[0:m].T) - dot(PSi, SIGMAi[:, 0:m])) * PHI[:, 0:m]

    dP = zeros((m, d))
    dG = zeros((m,d))

    for j in range(m):
        Delta = X[training, :] - P[j, :]
        dP[j, :] = dot(dlnPHI[:,j].T,Delta)*GAMMA[j,:]**2
        dG[j,:] = -diag(dot(GAMMA[j,:].reshape(d,1)*(Delta*dlnPHI[:,j].reshape(n, 1)).T,Delta))

    if(heteroscedastic):
        grad = concatenate((dP.flatten(), dG.flatten(), dlnAlpha.flatten(), db.flatten(), du.flatten(),dlnEta.flatten()))
    else:
        grad = concatenate((dP.flatten(), dG.flatten(), dlnAlpha.flatten(), db.flatten()))

    return -grad/n

def callbackF(theta):
    global iter
    global attempts

    global best_theta
    global best_valid

    global trainRMSE
    global trainLL

    global validRMSE
    global validLL

    global nlogML

    if ('iter' not in globals()):
        if 'validLL' not in globals():
            print  '{0:4s}\t{1:9s}\t\t{2:9s}\t\t{3:9s}'.format('Iter', ' logML/n', ' trainRMSE', ' trainLL')
        else:
            print  '{0:4s}\t{1:9s}\t\t{2:9s}\t\t{3:9s}\t\t{4:9s}\t\t{5:9s}'.format('Iter', ' logML/n', ' trainRMSE',
                                                                               ' trainRMSE/n', ' validRMSE', ' validLL')
        iter = 1

    if 'validLL' not in globals():
        print '{0:4d}\t{1: 1.7e}\t{2: 1.7e}\t{3: 1.7e}'.format(iter, -nlogML, trainRMSE,trainLL)
        best_theta = theta
        best_valid = trainLL
    else:

        if 'best_valid' not in globals() or validLL >= best_valid:
            print '{0:4d}\t{1: 1.7e}\t{2: 1.7e}\t{3: 1.7e}\t{4: 1.7e}\t[{5: 1.7e}]'.format(iter, -nlogML, trainRMSE, trainLL, validRMSE, validLL)
            best_theta = theta
            best_valid = validLL
            attempts = 0
        else:
            print '{0:4d}\t{1: 1.7e}\t{2: 1.7e}\t{3: 1.7e}\t{4: 1.7e}\t {5: 1.7e}'.format(iter, -nlogML, trainRMSE, trainLL, validRMSE, validLL)
            attempts = attempts+1

    iter += 1


def Dxy(X, Y):
    x2 = sum(power(X, 2), 1)
    y2 = sum(power(Y, 2), 1).T
    xy = dot(X, Y.T)
    D = add(add(y2, -2 * xy).T, x2).T

    return D


def split(n, trainSplit, validSplit, testSplit):
    r = random.permutation(n)

    validSample = ceil(n * validSplit)
    testSample = ceil(n * testSplit)
    trainSample = min([ceil(n * trainSplit), n - testSample - validSample])

    return sample(n,trainSample, validSample, testSample)


def sample(n, trainSample, validSample, testSample):
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