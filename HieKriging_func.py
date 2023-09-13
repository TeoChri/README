import matplotlib.pyplot as plt
from Kriging_func import Kriging
from Kernel_func import Rcompute
import scipy.linalg as sla
import numpy.linalg as la
import scipy.optimize as op

import numpy as np
import copy
from sklearn.gaussian_process.kernels import RBF

class MultiFidelityModel:
    eps = 1.e-10  # Cutoff for comparison to zero

    ### hyperParams = [ kernel_params]
    def __init__(self,
                 X_l,
                 Y_l,
                 X_h,
                 Y_h,
                 dim: int = None,
                 Rtype_h=1, Rtype_l=1):
        self.mu = None
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


        self.alpha = None
        self.invK = None
        self.rou = None

    #
    # def xNorm(self, x):
    #     x = x.reshape(-1, self.dim)
    #
    #     self.xlb = np.min(x, axis=0)
    #     self.xub = np.max(x, axis=0)
    #     self.scale = self.xub - self.xlb
    #
    #     res =
    #     return

    # def yNorm(self, y):
    #     self.ylb = np.min(y, axis=0)
    #     self.yub = np.max(y, axis=0)
    #     self.yscale = self.yub - self.ylb
    #
    #     res = self.yscale *
    #
    #     return
    #
    def likelihood_hf(self, theta_P):  # hyperParams = [ kernel_params]
        '''
        .. math::
            - log p(y|X) = 1/2 * y.T*K(X,X)^(-1)*y + 1/2 * log(|K|) + N/2 * log(2pi)

            X = [ X_LF       y = [ y_LF,       K(X,X) =  [ K_LL  , K_LH
                  X_HF ]           y_HF ]                  K_HL  , K_HH ]

            K_LL = K(X_LF, X_LF; hyp_LF)
            K_LH = rou * K(X_LF, X_HF; hyp_LF)
            K_HL = rou * K(X_HF, X_LF; hyp_LF)
            K_HH = rou^2 * K(X_HF, X_HF); hyp_LF) + K(X_HF, X_HF); hyp_HF)

            N = n_LF + n_HF

            y = [yLF,
                yHF]
            yLF = yTrainLF - fLF_pce(xTrainLF)
            yHF = yTrainHF - rou * yTrainLF - fHF_pce(xTrainLF)

            fLF = fLF_pce + fLF_gp
            fHF = rou * fLF + fHF_pce + fHF_gp
        :param hyp:
            array type, [sigma_eps, rou,  kernel_params]
        :return:
            array type, value of negtive log marginal likelihood
        '''

        '''1. Initialize the parameters.'''
        self.theta_P_h = theta_P
        n_l = self.X_l.shape[0]
        n_h = self.X_h.shape[0]

        xLf = self.X_l
        yLf = self.Y_l - self.modelLf.meanPredict(self.X_l)


        '''2. Compute the scale factor: rou.'''
        k_success = False
        try:
            ## LU Decomposition
            k = Rcompute(X1=self.X_h, X2=self.X_h, theta_P=self.theta_P_h, type=self.Rtype_h)
            p, l, u = sla.lu(k)
            inv_k = la.inv(u) @ (la.inv(l) @ p.T)
            k_success = True
        except:
            times = 0
            k = Rcompute(X1=self.X_h, X2=self.X_h, theta_P=self.theta_P_h, type=self.Rtype_h)
            while (not k_success) and (times < 3):
                k = k + np.eye(len(k)) * self.eps
                times += 1
                try:
                    ## LU Decomposition
                    p, l, u = sla.lu(k)
                    inv_k = la.pinv(u) @ (la.pinv(l) @ p.T)
                    k_success = True
                except:
                    pass

        # if k_success:
        #     F = self.modelLf.meanPredict(xHf)
        #     self.rou = F.T @ (inv_k*hypHf[0]) @ F / (F.T @ (inv_k*hypHf[0]) @ self.Y_h)
        F = self.meanLowPredict(self.X_h)
        if k_success:
            # F = F - np.mean(F)
            # yHf = yHf - np.mean(yHf)

            F_r_F = F.T @ (inv_k*hypHf[0]) @ F
            F_r_y = F.T @ (inv_k*hypHf[0]) @ yHf
            self.rou = F_r_y / F_r_F
        else:
            try:
                self.rou = np.mean(F, axis=0)/np.mean(yHf, axis=0)
            except:
                self.rou = 1


        '''3. Initialize the PCE model for MF part.'''
        # self.pceHf = PCE(xHf, self.Y_h - self.rou * self.modelLf.meanPredict(xHf), self.total_degree_hf)
        # self.kriHF = Kriging(xTrain=xHf,
        #                      yTrain=self.Y_h - self.rou * self.modelLf.meanPredict(xHf),
        #                      dim=self.dim,
        #                      kernelName=self.kernel_name_hf,
        #                      optMethod=self.opt_method_hf, )

        '''4. Compute the Cov matrix.'''
        n = len(xLf)
        m = len(xHf)
        N = n + m

        K_nn = kernel.k(xLf, xLf, hypLf, kernLf)
        K_mm = self.rou ** 2 * kernel.k(xHf, xHf, hypLf, kernLf) \
               + kernel.k(xHf, xHf, hypHf, kernHf)
        K_nm = self.rou * kernel.k(xLf, xHf, hypLf, kernLf)
        K_mn = self.rou * kernel.k(xHf, xLf, hypLf, kernLf)

        K_nN = np.hstack((K_nn, K_nm))
        K_mN = np.hstack((K_mn, K_mm))
        K_NN = np.vstack((K_nN, K_mN))
        K_NN = K_NN + np.eye(N) * self.eps

        ''' 5. Inverse the cov matrix.'''
        try:
            # LU Decomposition
            P, L, U = sla.lu(K_NN)
            self.invK = la.inv(U) @ (la.inv(L) @ P.T)
        except:
            success = False
            while not success:
                K_NN = K_NN + np.eye(N) * self.eps
                try:
                    P, L, U = sla.lu(K_NN)
                    self.invK = la.pinv(U) @ (la.inv(L) @ P.T)
                    success = True
                except:
                    NLML = 1000
                    return NLML

        '''6. Compute Negtive Log Maiginal Likelihood.'''
        yd = self.Y_h - self.rou * self.modelLf.meanPredict(xHf)

        # y = np.vstack((yLf, yd))
        # self.alpha = self.invK @ y
        #
        # detK = abs(np.prod(np.diag(U)))
        # NLML = y.T @ self.invK @ y \
        #        + np.log(detK)

        # y = np.vstack((self.rou * yLf, yHf))

        # self.mu = np.mean(y, axis=0)
        # self.mu = self.modelLf.meanPredict(np.vstack((xLf, xHf))) * self.rou
        # self.alpha = self.invK @ (y-self.mu)
        # self.alpha = self.invK @ y
        #
        # detK = abs(np.prod(np.diag(U)))
        # NLML = y.T @ self.invK @ y \
        #        + np.log(detK)

        detK = abs(np.prod(np.diag(u)))
        NLML = yd.T @ inv_k @ yd \
               + np.log(detK)

        x = np.vstack((xLf,xHf))
        y = np.vstack((yLf, yHf))
        # self.F = self.rou * self.meanLowPredict(x)
        self.F = np.vstack((self.meanLowPredict(xLf), self.rou*self.meanLowPredict(xHf)))
        self.FCF = self.F.T @ self.invK @ self.F
        self.alpha = self.invK @ (y-self.F)
        # self.alpha = self.invK @ y

        # f = np.ones((N, 1))
        # self.mu = f.T @ self.invK @ y / f.T @ self.invK @ f
        # self.alpha = self.invK @ (y - self.mu)

        # detr = abs(np.prod(np.diag(u)))
        # NLML = yd.T @ inv_k @ yd \
        #        + np.log(detr)



        return NLML

    def fitModel(self):
        """
        1. Fit the LF PCK model.
        2. Optimize the hyperparams of the HF part and rou.
        3. Inintial the PCE of the HF part.
        """

        '1. Fit the LF PCK model.'
        self.modelLf = Kriging(xTrain=self.X_l,
                             yTrain=self.Y_l,
                             dim=self.dim,
                             kernelName=self.kernel_name_hf,
                             optMethod=self.opt_method_hf, )

        # a = np.linspace(0, 1, 1000).reshape(-1, 1)
        # pp = self.modelLf
        # p = pp.pce
        # k = pp.kriging
        # plt.plot(a, pp.meanPredict(a), label='PCK')
        # plt.plot(a, k.meanPredict(a), label='GP')
        # plt.plot(a, p(a), label='PCE')
        # plt.fill_between(a.ravel(),
        #                  (pp.meanPredict(a) + 3 * k.stdPredict(a)).ravel(),
        #                  (pp.meanPredict(a) - 3 * k.stdPredict(a)).ravel(),
        #                  alpha=0.3,
        #                  label='confidence')
        # plt.scatter(self.X_l,
        #             self.Y_l)
        # plt.legend()
        # plt.title('PCK Test')
        # plt.show()

        '2. Optimize the hyperparams of the HF part and rou.'

        inihyp = copy.deepcopy(self.hyperParamsHf)
        iniNLNL = self.likelihood_hf(inihyp)

        hyp_best = copy.deepcopy(inihyp)
        bnd = self.bnd.tolist()
        # bnd = []
        # for i in range(len(inihyp)):
        #     bnd.append((1e-5, 100))
        for i in range(self.opt_times_hf):
            Result = op.minimize(fun=self.likelihood_hf,
                                 x0=hyp_best,
                                 method=self.opt_method_hf,
                                 bounds=bnd,
                                 options={'disp': False})  # jac=self.Gradient,

            hyp_best = copy.deepcopy(Result.x) \
                if Result.fun < iniNLNL else copy.deepcopy(inihyp)
            NLML_best = copy.deepcopy(Result.fun) \
                if Result.fun < iniNLNL else copy.deepcopy(iniNLNL)

        # from sko.GA import GA
        # ga = GA(self.likelihood_hf,
        #         len(inihyp),
        #         size_pop=50,
        #         max_iter=300,
        #         lb=self.bnd[:, 0],
        #         ub=self.bnd[:, 1])
        # hyp_best, NLML_best = ga.run()
        try:
            self.hyperParamsHf = copy.deepcopy(hyp_best) \
                if NLML_best < iniNLNL else copy.deepcopy(inihyp)

            NLML_best = self.likelihood_hf(self.hyperParamsHf)
            # print('MF likelihood is %d' % NLML_best)
            # print('MF paras are ' + str(self.hyperParamsHf))
        except:
            pass

        '3. Inintial the PCE of the HF part.'
        # rou = self.rou
        # xHf = self.X_h
        # yHf = self.Y_h - rou * self.modelLf.meanPredict(xHf)
        # self.pceHf = PCE(xHf, yHf, self.total_degree_hf)

        # a = np.linspace(0, 1, 1000).reshape(-1, 1)
        # pp = self.modelLf
        # # p = pp.pce
        # # k = pp.kriging
        # # plt.plot(a, self.pceHf(a), label='PCEHF')
        # plt.plot(a, pp.meanPredict(a), label='GP')
        # # plt.plot(a, p(a), label='PCE')
        # plt.fill_between(a.ravel(),
        #                  (pp.meanPredict(a) + 3 * pp.stdPredict(a)).ravel(),
        #                  (pp.meanPredict(a) - 3 * pp.stdPredict(a)).ravel(),
        #                  alpha=0.3,
        #                  label='confidence')
        # plt.scatter(self.X_l,
        #             self.Y_l)
        # plt.scatter(self.X_h,
        #             self.Y_h,
        #             label='HF sample')
        # plt.legend()
        # plt.title('PCK Test')
        # plt.show()
        return

    def meanPredict(self, xLocation):
        '''
        .. math::
            y_star_MF = rou * y_star_LF + f_PCE_MF(x_star) + f_kriging(x_star)
        :param xLocation:
            xstar
        :return:
            y_star_MF
        '''
        rou = self.rou
        xLocation = xLocation.reshape(-1, self.dim)

        xLf = self.X_l
        hypLf = self.modelLf.hyperParams
        kernLf = self.modelLf.kernelName

        xHf = self.X_h
        hypHf = self.hyperParamsHf
        kernHf = self.kernel_name_hf

        psiLf = rou * kernel.k(xLocation, xLf, hypLf, kernLf)
        psiHf = rou ** 2 * kernel.k(xLocation, xHf, hypLf, kernLf) + kernel.k(xLocation, xHf, hypHf, kernHf)
        psi = np.hstack((psiLf, psiHf))
        # psi = k(xLocation, xHf, hypHf, kernHf)

        mean_star = psi @ self.alpha
        # mean_star = psi @ self.invK @
        mean = mean_star + rou * self.modelLf.meanPredict(xLocation) #+ self.mu
        # mean = mean_star


        # plt.plot(xLocation, mean_star)
        # plt.scatter(self.X_h, self.Y_h - (rou * self.modelLf.pce(self.X_h) + self.pceHf(self.X_h)))
        # plt.title('MF PCK test.')
        # plt.show()
        return mean

    def varPredict(self, xLocation, full_cov: bool = False):
        """
        .. math::
            cov matrix = K_HH(x_star, x_star) - q.T * K(X, X)^{-1} * q

            K_HH(x_star, x_star) = K(x_star, x_star; hyp_HF)

            q.T = [  K_HL(x_star, x_LF)   ,   K_HH(x_star, x_HF)  ]
            K_HL(x_star, x_LF) = rou   * K(x_star, x_LF ; hyp_LF)
            K_HH(x_star, x_HF) = rou^2 * K(x_star, x_HF; hyp_LF) + K(x_star, x_HF; hyp_HF)
        :param xLocation:
            x_star
        :return:
            var
        """
        xLocation = xLocation.reshape(-1, self.dim)

        xLf = self.X_l
        hypLf = self.modelLf.hyperParams
        kernLf = self.modelLf.kernelName

        xHf = self.X_h
        hypHf = self.hyperParamsHf
        kernHf = self.kernel_name_hf
        rou = self.rou

        if full_cov:

            K_star_star = rou ** 2 * kernel.k(xLocation, xLocation, hypLf, kernLf) \
                          + kernel.k(xLocation, xLocation, hypHf, kernHf)

            psiLf = rou * kernel.k(xLocation, xLf, hypLf, kernLf)
            psiHf = rou ** 2 * kernel.k(xLocation, xHf, hypLf, kernLf) + kernel.k(xLocation, xHf, hypHf, kernHf)
            psi = np.hstack((psiLf, psiHf))

            # K_star_star = k(xLocation, xLocation, hypHf, kernHf)
            # psi = k(xLocation, xHf, hypHf, kernHf)
            u = self.F.T @ self.invK @ psi.T - self.meanLowPredict(xLocation)*self.rou
            # calculate prediction
            var_star_all = K_star_star - psi @ (self.invK @ psi.T) + u.T @ u / self.FCF
        else:
            # divided matrix calculation
            for i in range(100):
                if (i == 99):
                    x_star = xLocation[int(len(xLocation) / 100) * i:, :]
                else:
                    x_star = xLocation[int(len(xLocation) / 100) * i:int(len(xLocation) / 100) * (i + 1), :]

                K_star_star = rou ** 2 * kernel.k(x_star, x_star, hypLf, kernLf) \
                              + kernel.k(x_star, x_star, hypHf, kernHf)

                psiLf = rou * kernel.k(x_star, xLf, hypLf, kernLf)
                psiHf = rou ** 2 * kernel.k(x_star, xHf, hypLf, kernLf) + kernel.k(x_star, xHf, hypHf, kernHf)
                psi = np.hstack((psiLf, psiHf))
                # K_star_star = k(x_star, x_star, hypHf, kernHf)
                # psi = k(x_star, xHf, hypHf, kernHf)

                # calculate prediction
                u = self.F.T @ self.invK @ psi.T - self.meanLowPredict(x_star) * self.rou
                var_star = K_star_star - psi @ (self.invK @ psi.T) + u.T @ u / self.FCF
                var_star = abs(np.diag(var_star))

                # Combine mean_star and var_star
                if i == 0:
                    var_star_all = var_star
                else:
                    var_star_all = np.append([var_star_all], [var_star])

            var_star_all = var_star_all.reshape(-1, 1)

        return var_star_all

        # return var_star_all + rou ** 2 * self.modelLf.varPredict(xLocation, full_cov=full_cov)

    # def cov_l_h(self, xLocation):
    #     xLocation = xLocation.reshape(-1, self.dim)
    #
    #     xLf = self.X_l
    #     hypLf = self.modelLf.kriging.hyperParams
    #     kernLf = self.modelLf.kriging.kernelName
    #
    #     xHf = self.X_h
    #     hypHf = self.hyperParamsHf
    #     kernHf = self.kernel_name_hf
    #     rou = self.rou
    #
    #     # divided matrix calculation
    #     for i in range(100):
    #         if (i == 99):
    #             x_star = xLocation[int(len(xLocation) / 100) * i:, :]
    #         else:
    #             x_star = xLocation[int(len(xLocation) / 100) * i:int(len(xLocation) / 100) * (i + 1), :]
    #
    #         K_l_h = rou  * k(x_star, x_star, hypLf, kernLf)
    #
    #         psiLf_l_X = k(x_star, xLf, hypLf, kernLf)
    #         psiHf_l_X = rou * k(x_star, xHf, hypLf, kernLf) + k(x_star, xHf, hypHf, kernHf)
    #         psi_l_X  = np.hstack((psiLf_l_X , psiHf_l_X ))
    #
    #         psiLf_h_X = rou * k(x_star, xLf, hypLf, kernLf)
    #         psiHf_h_X = rou**2 * k(x_star, xHf, hypLf, kernLf) + k(x_star, xHf, hypHf, kernHf)
    #         psi_h_X = np.hstack((psiLf_h_X , psiHf_h_X ))
    #
    #         # calculate prediction
    #         cov_l_h = K_l_h - psi_l_X  @ (self.invK @ psi_h_X.T)
    #         # aaa = K_l_h - psi_h_X  @ (self.invK @ psi_l_X.T)
    #         cov_l_h = abs(np.diag(cov_l_h))
    #
    #         # Combine mean_star and var_star
    #         if i == 0:
    #             cov_l_h_all = cov_l_h
    #         else:
    #             cov_l_h_all = np.append([cov_l_h_all], [cov_l_h])
    #         cov_l_h_all = cov_l_h_all.reshape(-1, 1)
    #     return cov_l_h_all

    def stdPredict(self, xLocation):
        var_star_all = self.varPredict(xLocation, full_cov=False)

        std_star_all = np.sqrt(var_star_all).reshape(-1, 1)
        return std_star_all

    def meanLowPredict(self, xLocation):
        """
        .. math::
            y_star_LF = f_PCK_LF(x_star)
        :param xLocation:
            xstar
        :return:
            y_star_LF
        """
        meanLf = self.modelLf.meanPredict(xLocation)
        np.asarray(meanLf).reshape(-1, 1)
        return meanLf

    def varLowPredict(self, xLocation, full_cov: bool = False):
        varLf = self.modelLf.varPredict(xLocation, full_cov=full_cov)
        varLf = np.asarray(varLf).reshape(-1, 1)
        return varLf

    def stdLowPredict(self, xLocation):
        stdLf = self.modelLf.stdPredict(xLocation)
        stdLf = np.asarray(stdLf).reshape(-1, 1)
        return stdLf

    def addSample(self, xNew, yNew, fidelity: str = 'LF'):
        xNew = xNew.reshape(len(yNew), -1)
        yNew = yNew.reshape(-1, 1)

        if fidelity == 'HF':
            self.X_h = np.vstack((self.X_h, xNew))
            self.Y_h = np.vstack((self.Y_h, yNew))
        elif fidelity == 'LF':
            self.X_l = np.vstack((self.X_l, xNew))
            self.Y_l = np.vstack((self.Y_l, yNew))

        self.fitModel()
