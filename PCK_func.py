# import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
from Kernel_func import Rcompute
# from PCE_func import PCE
from PCE_Chaospy_func import PCE

class PCK():
    eps = 1e-14

    def __init__(self, X, Y, dim=None, Rtype=1):
        if dim is None:
            self.X = np.atleast_2d(X)
            self.Y = np.atleast_2d(Y)

            self.n = self.X.shape[0]
            self.dim = self.X.shape[1]
        else:
            self.X = X.reshape(-1, dim)
            self.n = self.X.shape[0]
            self.dim = dim

            self.Y = Y.reshape(self.n, -1)

        self.Rtype = Rtype
        self.theta_P = None
        # lbX = np.zeros((1, self.dim))
        # ubX = np.oneos((1, self.dim))

    def likelihood(self, theta_P):
        self.R = Rcompute(X1=self.X, X2=self.X, theta_P=theta_P, type=self.Rtype)

        try:
            self.invR = np.linalg.pinv(self.R + self.eps * np.eye(len(self.R)))
            # one = np.ones((self.n, 1))
            # self.mu = (one.T @ self.invR @ self.Y) / (one.T @ self.invR @ one)
            self.mu = self.PCE(self.X)

            ydelta = self.Y - self.mu
            self.sigma2 = (ydelta.T) @ self.invR @ ydelta / self.n
            self.alpha = self.invR @ ydelta

            if self.sigma2 < 0:
                self.sigma2 = 1e-12

            detR = np.linalg.det(self.R)
            NLML = np.log(detR) + self.n * np.log(self.sigma2)
        except:
            NLML = 1e4

        return NLML

    def fitModel(self, optMethod='Nelder-Mead'):
        self.PCE = PCE(self.X, self.Y)

        nTheta = self.dim
        if self.Rtype == 1:
            nP = self.dim
        else:
            nP = 0
        lbTheta_P = np.hstack((1e-5 * np.ones((1, nTheta)), 1.01 * np.ones((1, nP))))
        ubTheta_P = np.hstack((1e3 * np.ones((1, nTheta)), 1.999 * np.ones((1, nP))))

        if self.theta_P is not None:
            theta_P0 = self.theta_P
        else:
            theta_P0 = np.hstack((1 * np.ones((1, nTheta)), 1.5 * np.ones((1, nP))))

        Result = op.minimize(fun=self.likelihood,
                             x0=theta_P0,
                             method=optMethod,
                             bounds=np.vstack((lbTheta_P, ubTheta_P)).T,
                             options={'disp': False})
        self.theta_P = Result.x

        NLML_best = self.likelihood(self.theta_P)
        return

    def meanPredict(self, x):
        x = x.reshape(-1, self.dim)

        r_xX = Rcompute(x, self.X, self.theta_P, self.Rtype)

        mean = self.PCE(x) + r_xX @ self.alpha
        return mean

    def varPredict(self, x, cov=False):
        x = x.reshape(-1, self.dim)

        r_xX = Rcompute(x, self.X, self.theta_P, self.Rtype)

        if cov:
            R_xx = Rcompute(x, x, self.theta_P, self.Rtype) + self.eps * np.eye(self.n)

        else:
            R_xx = np.eye(len(x)) * (1 + self.eps)

        one = np.ones((self.n, 1))
        Cov_xx = self.sigma2 * (R_xx - r_xX @ self.invR @ r_xX.T)
        Cov_xx = Cov_xx + (1 - r_xX @ self.invR @ r_xX.T) ** 2 / (one.T @ self.invR @ one)

        return Cov_xx if cov else np.diag(Cov_xx).reshape(-1, 1)

    def stdPredict(self, x):
        var = self.varPredict(x, 0)
        std = np.sqrt(var).reshape(-1, 1)
        return std

    def addSample(self, xNew, yNew):
        xNew = xNew.reshape(-1, self.dim)
        yNew = yNew.reshape(len(xNew), -1)

        self.X = np.vstack((self.X, xNew))
        self.Y = np.vstack((self.Y, yNew))

        self.fitModel()
