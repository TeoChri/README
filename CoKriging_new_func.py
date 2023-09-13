# import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
from Kernel_func import Rcompute
from Kriging_func import Kriging


class CoKriging():
    eps = 1e-14

    def __init__(self, X_l, Y_l, X_h, Y_h, dim=None, Rtype_h=1, Rtype_l=1):
        if dim is None:
            self.X_l = np.atleast_2d(X_l)
            self.X_h = np.atleast_2d(X_h)
            self.Y_l = np.atleast_2d(Y_l)
            self.Y_h = np.atleast_2d(Y_h)

            self.dim = self.X_l.shape[1]
        else:

            self.X_l = X_l.reshape(-1, dim)
            self.X_h = X_h.reshape(-1, dim)

            n_l = self.X_l.shape[0]
            n_h = self.X_h.shape[0]
            self.Y_l = Y_l.reshape(n_l, -1)
            self.Y_h = Y_h.reshape(n_h, -1)

            self.dim = dim

        self.Rtype_h = Rtype_h
        self.Rtype_l = Rtype_l
        self.theta_P_h = None
        # lbX = np.zeros((1, self.dim))
        # ubX = np.oneos((1, self.dim))

    # def likelihood(self, theta_P_rho):
    #     self.theta_P_h = theta_P_rho[:-1]
    #     self.rho = theta_P_rho[-1]

    def likelihood(self, theta_P):
        self.theta_P_h = theta_P
        n_l = self.X_l.shape[0]
        n_h = self.X_h.shape[0]

        self.R = Rcompute(X1=self.X_h, X2=self.X_h, theta_P=self.theta_P_h, type=self.Rtype_h)
        try:
            invR = np.linalg.pinv(self.R + self.eps * np.eye(n_h))

            yh_l = self.meanLowPredict(self.X_h)
            self.rho = (yh_l.T @ invR @ self.Y_h) / ((yh_l.T @ invR @ yh_l))

            yd = self.Y_h - self.rho * yh_l
            one = np.ones((n_h, 1))
            mu = (one.T @ invR @ yd) / (one.T @ invR @ one)

            ydelta = yd - mu
            self.sigma2 = ydelta.T @ invR @ ydelta / n_h

            if self.sigma2 < 0:
                self.sigma2 = 1e-12

            detR = np.linalg.det(self.R)
            NLML = np.log(detR) + n_h * np.log(self.sigma2)
        except:
            NLML = 1e4


        return NLML

    def fitModel(self, optMethod='Nelder-Mead'):
        '1.建立低精度模型'
        self.model_l = Kriging(self.X_l, self.Y_l, dim=self.dim, Rtype=self.Rtype_l)
        self.model_l.fitModel()

        self.model_h = Kriging(self.X_h, self.Y_h, dim=self.dim, Rtype=self.Rtype_h)
        self.model_h.fitModel()
        '2.优化修正模型的参数'
        if self.Rtype_h == 1:
            nTheta = self.dim
            nP = self.dim
        elif self.Rtype_h == 6:
            nTheta = 1
            nP = 1
        else:
            nTheta = self.dim
            nP = 0

        # lbTheta_P_rho = np.hstack((1e-5 * np.ones((1, nTheta)), 1.01 * np.ones((1, nP)), [[10]]))
        # ubTheta_P_rho = np.hstack((1e3 * np.ones((1, nTheta)), 1.999 * np.ones((1, nP)), [[10]]))
        #
        # if self.theta_P_h is not None and self.rho is not None:
        #     theta_P_rho0 = np.hstack((self.theta_P_h, self.rho))
        # else:
        #     theta_P_rho0 = np.hstack((1 * np.ones((1, nTheta)), 1.5 * np.ones((1, nP)), [[1]]))
        #
        # Result = op.minimize(fun=self.likelihood,
        #                      x0=theta_P_rho0,
        #                      method=optMethod,
        #                      bounds=np.vstack((lbTheta_P_rho, ubTheta_P_rho)).T,
        #                      options={'disp': False})
        # self.theta_P_h = Result.x[-1:]
        # self.rho = Result.x[-1]

        lbTheta_P = np.hstack((1e-5 * np.ones((1, nTheta)), 1.01 * np.ones((1, nP))))
        ubTheta_P = np.hstack((1e3 * np.ones((1, nTheta)), 1.999 * np.ones((1, nP))))

        if self.theta_P_h is not None and self.rho is not None:
            theta_P = np.hstack((self.theta_P_h))
        else:
            theta_P = np.hstack((1 * np.ones((1, nTheta)), 1.5 * np.ones((1, nP))))

        Result = op.minimize(fun=self.likelihood,
                             x0=theta_P,
                             method=optMethod,
                             bounds=np.vstack((lbTheta_P, ubTheta_P)).T,
                             options={'disp': False})
        self.theta_P_h = Result.x
        NLML_best = self.likelihood(Result.x)

        '3.计算总协方差方程'
        rho = self.rho
        n_l = self.X_l.shape[0]
        n_h = self.X_h.shape[0]

        s2_h = self.sigma2
        s2_l = self.model_l.sigma2

        Rl_ll = self.model_l.R
        Rl_hl = Rcompute(self.X_h, self.X_l, self.model_l.theta_P, type=self.Rtype_l)
        Rl_lh = Rl_hl.T
        Rh_hh = self.R
        Rl_hh = Rcompute(self.X_h, self.X_h, self.model_l.theta_P, type=self.Rtype_l)

        self.C = np.vstack((np.hstack((s2_l * Rl_ll, rho * s2_l * Rl_lh)),
                            np.hstack((rho * s2_l * Rl_hl, rho ** 2 * s2_l * Rl_hh + s2_h * Rh_hh))))
        self.invC = np.linalg.pinv(self.C + self.eps * np.eye(len(self.C)))

        Y = np.vstack((self.Y_l, self.Y_h))
        one = np.ones((n_h + n_l, 1))
        self.mu = (one.T @ self.invC @ Y) / (one.T @ self.invC @ one)
        Ydelta = Y - self.mu
        self.alpha = self.invC @ Ydelta
        return

    def meanPredict(self, x):
        x = x.reshape(-1, self.dim)

        rho = self.rho
        s2_l = self.model_l.sigma2
        s2_h = self.sigma2

        rl_xXl = Rcompute(x, self.X_l, self.model_l.theta_P, type=self.Rtype_l)
        rl_xXh = Rcompute(x, self.X_h, self.model_l.theta_P, type=self.Rtype_l)
        rh_xXh = Rcompute(x, self.X_h, self.theta_P_h, type=self.Rtype_h)
        c_xX = np.hstack((rho * s2_l * rl_xXl,
                          rho**2 * s2_l * rl_xXh + s2_h * rh_xXh))

        mean = self.mu + c_xX @ self.alpha
        return mean

    def varPredict(self, x, cov=False):
        x = x.reshape(-1, self.dim)

        rho = self.rho


        Cov_xx = rho ** 2 * self.model_l.varPredict(x, cov)\
                 + self.model_h.varPredict(x, cov)

        return Cov_xx

    def stdPredict(self, x):
        var = self.varPredict(x, 0)
        std = np.sqrt(var).reshape(-1, 1)
        return std

    def meanLowPredict(self,x):
        mean_l = self.model_l.meanPredict(x)
        return mean_l

    def varLowPredict(self, x, cov=False):
        var_l = self.model_l.varPredict(x, cov)
        return var_l

    def stdLowPredict(self, x):
        std_l = self.model_l.stdPredict(x)
        return std_l

    def addSample(self, xNew, yNew, fidelity='l'):
        xNew = xNew.reshape(-1, self.dim)
        yNew = yNew.reshape(len(xNew), -1)

        if fidelity == 'l' or fidelity == 'LF':
            self.X_l = np.vstack((self.X_l, xNew))
            self.Y_l = np.vstack((self.Y_l, yNew))
        elif fidelity == 'h' or fidelity == 'HF':
            self.X_h = np.vstack((self.X_h, xNew))
            self.Y_h = np.vstack((self.Y_h, yNew))
        else:
            print('Sample is not added.')

        self.fitModel()
