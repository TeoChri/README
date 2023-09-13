import functools as ft
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.utils.extmath import cartesian
import copy
import pandas as pd

from Test_func import RMSE, MAE, Rsquare, MAPE
from BO_compute_func import Opt_Random_Sample, mes_compute, dmes_compute, ei_compute, efi_compute
from benchmark_func import *

from CoKriging_func import CoKriging as MF
# from CoPCK_func import CoPCK as MF
from expRAE_func import expResult, expAllResult
from expNACA import expNACA, expAllNACA


# from Kernel_func import kernelFunc
# kernel = kernelFunc()

class MF_BO:
    def __init__(self,
                 dir,
                 input_dim,
                 benchmark_name,
                 grid_size,
                 x_pool,
                 x_train_l,
                 x_train_h,
                 af_name: str,
                 seed: int = 0,
                 cost: int = 10,
                 maximize: bool = True
                 ):
        """
        1. Initialize the benchmark function.
        2. Initialize the MF model.
        3. Initialize the parameters for BO.
        4. Ready for testing the model performance.
        5. Ready for sequential sampling.
        """
        self.dir = dir
        '''1. Initialize the benchmark function.'''
        self.input_dim = input_dim
        self.benchmark_name = benchmark_name

        if benchmark_name == 'RAE2822':
            self.f_h = ft.partial(expResult, fidelity='HF', grid_size=grid_size)
            self.f_l = ft.partial(expResult, fidelity='LF', grid_size=grid_size)
        elif benchmark_name == 'NACA':
            i_loc = 157
            self.f_h = ft.partial(expNACA, fidelity='HF', i_loc=i_loc)
            self.f_l = ft.partial(expNACA, fidelity='LF', i_loc=i_loc)
        else:
            self.f_h_compute = {'Forrester 1a': FO,
                                'Forrester 1b': FO,
                                'Forrester 1c': FO,
                                'Gramacy-Lee': GL,
                                'Currin': Currin,
                                'Bukin': Bukin,
                                'Branin': Branin,
                                'Six-hump camel-back': SC,
                                'Booth': Booth,
                                'Bohachevsky': Bohachevsky,
                                'Park': Park,
                                'Park2': Park2,
                                'Borehole': Borehole,
                                'Sasena': Sasena,
                                }
            assert benchmark_name in self.f_h_compute.keys(), \
                f'Unknown benchmark function: {benchmark_name}'

            self.f_l_compute = {'Forrester 1a': FOa_low,
                                'Forrester 1b': FOa_low,
                                'Forrester 1c': FOa_low,
                                'Gramacy-Lee': GL_low,
                                'Currin': Currin_low,
                                'Bukin': Bukin_low,
                                'Branin': Branin_low,
                                'Six-hump camel-back': SC_low,
                                'Booth': Booth_low,
                                'Bohachevsky': Bohachevsky_low,
                                'Park': Park_low,
                                'Park2': Park2_low,
                                'Borehole': Borehole_low,
                                'Sasena': Sasena_low,
                                }
            assert benchmark_name in self.f_l_compute.keys(), \
                f'Unknown benchmark function: {benchmark_name}'

            self.f_h = self.f_h_compute[self.benchmark_name]
            self.f_l = self.f_l_compute[self.benchmark_name]

        '''2. Initialize the MF model.'''
        # grid_1d = np.linspace(0, 1, grid_size)
        # x_pool = cartesian([grid_1d for _ in range(input_dim)])
        self.x_pool = x_pool

        if benchmark_name == 'RAE2822':
            self.y_pool = expAllResult(fidelity='HF', grid_size=grid_size)
            self.y_pool_l = expAllResult(fidelity='LF', grid_size=grid_size)
        elif benchmark_name == 'NACA':
            self.y_pool = expAllNACA(fidelity='HF', i_loc=i_loc)
            self.y_pool_l = expAllNACA(fidelity='LF', i_loc=i_loc)
        else:
            self.y_pool = self.f_h(self.x_pool)
            self.y_pool_l = self.f_l(self.x_pool)

        if benchmark_name == 'NACA':
            self.idx_l = grid_size[0]
            self.idx_h = grid_size[1]
            y_train_h = self.f_h(np.squeeze(self.idx_h))
            y_train_l = self.f_l(np.squeeze(self.idx_l))

        else:
            y_train_h = self.f_h(x_train_h)
            y_train_l = self.f_l(x_train_l)
        self.model = MF(x_train_l,
                        y_train_l,
                        x_train_h,
                        y_train_h,
                        dim=input_dim,
                        Rtype_h=11,
                        Rtype_l=11
                        )
        self.model.fitModel()

        # self.model.varPredict(self.model.X_h[1])

        '''3. Initialize the parameters for BO.'''
        self.af_name = af_name
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.cost = cost
        self.maximize = maximize
        self.iteration = 0

        self.af_compute = {'DMES': self.dmes,
                           'VFEI': self.vfei,
                           # 'random': self.random,
                           'EFI': self.efi,
                           'EI': self.ei,
                           'MES': ft.partial(self.mes, n_sample=100),
                           'VFMES': self.vfmes,
                           'K-MES':ft.partial(self.mes, n_sample=100),
                           }
        # af_compute = {'pi', 'ei', 'ucb', 'gp-ucb', 'random', 'gp-ucb', 'mes'}
        assert af_name in self.af_compute.keys(), f'Unknown acquisition function: {af_name}'

        '''4. Ready for testing the model performance.'''

        self.y_opt = np.max(self.y_pool) if maximize else np.min(self.y_pool)
        self.y_best = np.max(self.model.Y_h) if maximize else np.min(self.model.Y_h)
        # self.y_best_set = None

        self.mean_h = None
        self.std_h = None
        # self.cov_h = None

        self.mean_l = None
        self.std_l = None

        self.simple_r = []
        self.inference_r = []
        self.rmse = []
        self.mae = []
        self.mape = []
        self.rsquare = []
        self.total_cost = []

        '''5. Ready for sequential sampling.'''
        self.next_x = None
        self.next_i = None
        self.next_x_af_value = None
        self.next_fidelity = None

        self.preCompute()
        return

    def next_observation(self, plot_perfomance=True, plot_af=True):
        """ """
        '''1. Calculate the acquisition function.
              Then, query the new observation.'''
        af_h, af_l = self.af_compute[self.af_name]()

        i = -1
        while abs(i) < len(af_h) and any(np.allclose(self.x_pool[np.argsort(af_h)[i]], self.model.X_h[j]) for j in
                                         range(len(self.model.X_h))):
            i = i - 1
        max_i_h = np.argsort(af_h)[i]

        i = -1
        while abs(i) < len(af_l) and any((np.allclose(self.x_pool[np.argsort(af_l)[i]], self.model.X_l[j]) for j in
                                          range(len(self.model.X_l)))):
            i = i - 1
        max_i_l = np.argsort(af_l)[i]

        if af_h[max_i_h] >= af_l[max_i_l]:
            self.next_fidelity = 'HF'

            max_i = max_i_h
            # self.next_i = max_i
            self.next_x = self.x_pool[max_i]
            self.next_x_af_value = af_h[max_i]
            # self.next_y = self.f_h(self.next_x)
            self.next_y = self.y_pool[max_i]
        elif af_h[max_i_h] < af_l[max_i_l]:
            # if af_l[max_i_l]>0:
            self.next_fidelity = 'LF'

            max_i = max_i_l
            # self.next_i = max_i
            self.next_x = self.x_pool[max_i]
            self.next_x_af_value = af_l[max_i]
            # self.next_y = self.f_l(self.next_x)
            self.next_y = self.y_pool_l[max_i]
            # else:
            #     self.next_fidelity = 'HF'
            #
            #     max_i = max_i_h
            #     # self.next_i = max_i
            #     self.next_x = self.x_pool[max_i]
            #     self.next_x_af_value = af_h[max_i]
            #     self.next_y = self.f_h(self.next_x)
        else:
            print('The calculation is wrong.')

        '''2. Show the detail of the result of acquisition function'''
        if plot_af:
            if self.input_dim == 1:
                plt.scatter(self.model.X_h,
                            np.zeros(len(self.model.X_h)),
                            label='HF samples : %d' % len(self.model.X_h))
                plt.scatter(self.model.X_l,
                            np.zeros(len(self.model.X_l)),
                            label='LF samples : %d' % len(self.model.X_l))
                plt.scatter(self.next_x,
                            self.next_x_af_value,
                            label='next sample')

                plt.plot(self.x_pool, af_h, label='HF')
                plt.plot(self.x_pool, af_l, label='LF')

                plt.xlabel('X')
                plt.ylabel('Acqusiton Function')

                plt.title('%s,  \n %s for %d itr'  # seed=%d
                          % (self.benchmark_name,
                             # self.seed,
                             self.af_name,
                             self.iteration))

            if self.input_dim == 2:
                from mpl_toolkits import mplot3d
                size = int(np.sqrt(len(self.x_pool)))
                X = self.x_pool[:, 0].reshape(size, size).T
                Y = self.x_pool[:, 1].reshape(size, size).T
                af_h_mesh = af_h.reshape(size, size).T
                af_l_mesh = af_l.reshape(size, size).T

                fig = plt.figure()  # figsize=(10,10))

                # plot the HF af
                ax1 = fig.add_subplot(1, 2, 1, projection='3d')
                ax1.contour3D(X, Y, af_h_mesh, 50)
                ax1.view_init(elev=28, azim=160)

                # plot the LF af
                ax2 = fig.add_subplot(1, 2, 2, projection='3d')
                ax2.contour3D(X, Y, af_l_mesh, 50)
                ax2.view_init(elev=28, azim=160)

                ax_list = [ax1, ax2]
                title_list = [r'$MES_h$', r'$MES_{h\l}$']
                for i in range(len(ax_list)):
                    ax = ax_list[i]
                    title = title_list[i]

                    ax.scatter3D(self.model.X_h[:, 0],
                                 self.model.X_h[:, 1],
                                 np.zeros(len(self.model.X_h)),
                                 label='HF samples : %d' % len(self.model.X_h)
                                 )
                    ax.scatter3D(self.model.X_l[:, 0],
                                 self.model.X_l[:, 1],
                                 np.zeros(len(self.model.X_l)),
                                 label='LF samples : %d' % len(self.model.X_l)
                                 )
                    ax.scatter3D(self.next_x[0],
                                 self.next_x[1],
                                 self.next_x_af_value,
                                 label='next sample',
                                 color='r',
                                 linewidths=4
                                 )

                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Acqusiton Function')
                    ax.set_title(title)

                plt.suptitle('%s,  \n %s for %d itr'  # seed=%d
                             % (self.benchmark_name,
                                # self.seed,
                                self.af_name,
                                self.iteration))

            if self.benchmark_name == 'NACA':
                plt.plot(af_h, label='HF')
                plt.plot(af_l, label='LF')
                if self.next_fidelity == 'HF':
                    self.idx_h = np.vstack((self.idx_h, max_i))
                else:
                    self.idx_l = np.vstack((self.idx_l, max_i))

            plt.legend()
            plt.savefig(self.dir + '%s_%s_seed%d_%d_S.png'
                        % (self.benchmark_name,
                           self.af_name,
                           self.seed,
                           self.iteration))
            # plt.show()
            plt.close()

        '''3.  Update the model. '''
        self.model.addSample(self.next_x,
                             self.next_y,
                             self.next_fidelity)

        '''4. Ready for next observation.'''
        self.iteration += 1
        self.preCompute(plot_perfomance)

        return

    def preCompute(self, plot_performance=True):
        self.mean_h = self.model.meanPredict(self.x_pool)
        self.std_h = self.model.stdPredict(self.x_pool)
        self.mean_l = self.model.meanLowPredict(self.x_pool)
        self.std_l = self.model.stdLowPredict(self.x_pool)

        ''' Test the model performance.'''
        self.testPerformance()
        self.saveResult()

        if plot_performance:
            if self.input_dim == 1:
                plt.scatter(self.model.X_h,
                            self.model.Y_h,
                            label='HF samples : %d' % len(self.model.X_h),
                            marker='o')  # , label='HF sampled points'
                plt.scatter(self.model.X_l,
                            self.model.Y_l,
                            label='LF samples : %d' % len(self.model.X_l),
                            marker='*')  # , label='LF sampled points'

                plt.plot(self.x_pool,
                         self.y_pool,
                         label='True HF model')
                plt.plot(self.x_pool,
                         self.y_pool_l,
                         label='True LF model')

                plt.plot(self.x_pool,
                         self.mean_h,
                         linestyle='-.',
                         # label='MF predict',
                         )
                plt.fill_between(np.squeeze(self.x_pool),
                                 np.squeeze(self.mean_h + 1 * self.std_h),
                                 np.squeeze(self.mean_h - 1 * self.std_h),
                                 alpha=0.3,
                                 label='MF model confidence')

                plt.plot(self.x_pool,
                         self.mean_l,
                         linestyle=':',
                         # label='LF model predict',
                         )
                plt.fill_between(np.squeeze(self.x_pool),
                                 np.squeeze(self.mean_l + 1 * self.std_l),
                                 np.squeeze(self.mean_l - 1 * self.std_l),
                                 alpha=0.3,
                                 label='LF model confidence')

                if self.next_fidelity == 'HF':
                    plt.scatter(self.next_x,
                                self.f_h(self.next_x),
                                label='New sample',
                                marker='^', linewidths=4, edgecolor='r')
                elif self.next_fidelity == 'LF':
                    plt.scatter(self.next_x,
                                self.f_l(self.next_x),
                                label='New sample',
                                marker='^', linewidths=4, edgecolor='r')
                else:
                    pass

                # from Kriging_func import Kriging
                # krig = Kriging(self.model.X_h, self.model.Y_h)
                # m = krig.meanPredict(self.x_pool)
                # s = krig.stdPredict(self.x_pool)
                # plt.plot(self.x_pool, m, label='GP predict')
                # plt.fill_between(np.squeeze(self.x_pool),
                #                  np.squeeze(m + 3 * s),
                #                  np.squeeze(m - 3 * s),
                #                  alpha=0.3,
                #                  label='GP confidence')

                # plt.ylim(-20, 20)
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.title('%s, %s \n model for %d itr, total cost:%s'  # seed=%d
                          % (self.benchmark_name,
                             self.af_name,
                             # self.seed,
                             self.iteration,
                             str(self.total_cost[self.iteration])))

                plt.legend()
                plt.savefig(self.dir + '%s_%s_seed%d_%d_Model.png'
                            % (self.benchmark_name,
                               self.af_name,
                               self.seed,
                               self.iteration))
                # plt.show()
                plt.close()

            if self.input_dim == 2:
                from mpl_toolkits import mplot3d
                size = int(np.sqrt(len(self.x_pool)))
                X = self.x_pool[:, 0].reshape(size, size).T
                Y = self.x_pool[:, 1].reshape(size, size).T
                z_h_mesh = self.y_pool.reshape(size, size).T
                z_l_mesh = self.y_pool_l.reshape(size, size).T

                #
                # ax = plt.figure().add_subplot(111, projection='3d')
                # ax.plot_surface(X, Y, z_h_mesh, cmap = plt.get_cmap('rainbow'),antialiased=True)
                # ax.plot_surface(X, Y, z_l_mesh, alpha=0.4, color='aqua')
                # ax.view_init(elev=40, azim=-50)
                # plt.show()
                # plt.savefig(self.dir + '%s_%s_seed%d_%d_Model.png'
                #             % (self.benchmark_name,
                #                self.af_name,
                #                self.seed,
                #                self.iteration,
                #                ))

                fig = plt.figure()
                ax1 = fig.add_subplot(2, 2, 1, projection='3d')
                # ax1.contour3D(X, Y, z_h_mesh, 50, label='HF')
                ax1.plot_surface(X, Y, z_h_mesh, alpha=0.8, cmap=plt.get_cmap('rainbow'), antialiased=True)
                ax1.plot_surface(X, Y, z_l_mesh, alpha=0.4, color='aqua')
                ax1.view_init(elev=28, azim=160)

                # ax2 = fig.add_subplot(2, 2, 2, projection='3d')
                # # ax2.contour3D(X, Y, z_l_mesh, 50)
                # ax2.plot_surface(X, Y, z_l_mesh, cmap = plt.get_cmap('rainbow'),antialiased=True)

                ax3 = fig.add_subplot(2, 2, 2, projection='3d')
                # ax3.contour3D(X, Y, self.mean_h.reshape(size, size).T, 50)
                ax3.plot_surface(X, Y, self.mean_h.reshape(size, size).T, alpha=0.8, cmap=plt.get_cmap('rainbow'))
                ax3.plot_surface(X, Y, self.mean_l.reshape(size, size).T, alpha=0.4, color='palegreen')
                ax3.view_init(elev=28, azim=160)

                # ax4 = fig.add_subplot(2, 2, 4, projection='3d')
                # # ax4.contour3D(X, Y, self.mean_l.reshape(size, size).T, 50)
                # ax4.plot_surface(X, Y, self.mean_l.reshape(size, size).T, rstride = 1, cstride = 1, cmap = plt.get_cmap('rainbow'))

                from Kriging_func import Kriging
                kri = Kriging(self.model.X_h, self.model.Y_h, self.input_dim)
                kri.fitModel()
                ax5 = fig.add_subplot(2, 2, 3, projection='3d')
                # ax3.contour3D(X, Y, self.mean_h.reshape(size, size).T, 50)
                ax5.plot_surface(X, Y, kri.meanPredict(self.x_pool).reshape(size, size).T, alpha=0.8,
                                 cmap=plt.get_cmap('rainbow'))
                ax5.view_init(elev=28, azim=160)

                ax6 = fig.add_subplot(2, 2, 4, projection='3d')
                # plt.contour(X, Y, self.std_h.reshape(size, size).T)
                # plt.scatter(self.model.X_h[:, 0],
                #                  self.model.X_h[:, 1],
                #                 )
                ax6.plot_surface(X, Y, self.std_h.reshape(size, size).T, alpha=0.8, cmap=plt.get_cmap('rainbow'))
                ax6.view_init(elev=28, azim=160)

                ax_list = [ax1, ax3, ax5]
                title_list = ['True Response', 'MF Model', 'SF model']
                # ax_list = [ax1, ax2, ax3, ax4]
                # title_list = ['HF model', 'LF model', 'MF predict', 'LF predict']
                for i in range(len(ax_list)):
                    ax = ax_list[i]
                    ax.scatter3D(self.model.X_h[:, 0],
                                 self.model.X_h[:, 1],
                                 self.model.Y_h,
                                 label='HF samples : %d' % len(self.model.X_h),
                                 marker='o',
                                 color='g'
                                 )
                    ax.scatter3D(self.model.X_l[:, 0],
                                 self.model.X_l[:, 1],
                                 self.model.Y_l,
                                 label='LF samples : %d' % len(self.model.X_l),
                                 marker='*',
                                 color='orange'
                                 )
                    try:
                        ax.scatter3D(self.next_x[0],
                                     self.next_x[1],
                                     self.next_y,
                                     label='new samples',
                                     marker='^',
                                     color='r',
                                     linewidths=4,
                                     )
                    except:
                        pass

                    ax.set_xlabel('AOA')
                    ax.set_ylabel('MACH')
                    ax.set_zlabel(r'$C_L$')
                    ax.set_title(title_list[i])
                    plt.grid(True)

                plt.suptitle('%s \\ model for %d itr, total cost:%s'
                             % (self.benchmark_name,
                                # self.seed,
                                self.iteration,
                                str(self.total_cost[self.iteration])))
                # plt.legend()
                plt.savefig(self.dir + '%s_%s_seed%d_%d_Model.png'
                            % (self.benchmark_name,
                               self.af_name,
                               self.seed,
                               self.iteration,
                               ))
                # plt.show()
                plt.close()

            if self.benchmark_name == 'RAE2822' and True:
                plt.plot(X[0, :], z_l_mesh[10], label='LF simulation')
                plt.plot(X[0, :], self.mean_l.reshape(size, size).T[10], label='LF model')
                plt.legend()
                plt.xlabel('AOA')
                plt.ylabel(r'$C_L$')
                plt.title('%s\\total cost=%s, LF samples:%d; HF samples:%d' % (self.af_name,
                                                                               str(self.total_cost[-1]),
                                                                               len(self.model.X_l),
                                                                               len(self.model.X_h)))
                plt.savefig(self.dir + '%s_AOA_l_%d.png'
                            % (
                                self.af_name,
                                self.iteration,
                            ))

                # plt.show()
                plt.close()

                plt.plot(X[0, :], z_h_mesh[1], label='HF simulation')
                plt.plot(X[0, :], self.mean_h.reshape(size, size).T[1], label='MF model')
                plt.legend()
                plt.xlabel('AOA')
                plt.ylabel(r'$C_L$')
                plt.title('%s\\total cost=%s, LF samples:%d; HF samples:%d' % (self.af_name,
                                                                               str(self.total_cost[-1]),
                                                                               len(self.model.X_l),
                                                                               len(self.model.X_h)))
                plt.savefig(self.dir + '%s_AOA_h_%d.png'
                            % (
                                self.af_name,
                                self.iteration,
                            ))

                # plt.show()
                plt.close()

                plt.plot(Y[:, 0], z_l_mesh[:, 19], label='LF similation')
                plt.plot(Y[:, 0], self.mean_l.reshape(size, size).T[:, 17], label='LF model')
                plt.legend()
                plt.xlabel('MACH')
                plt.ylabel(r'$C_L$')
                plt.title('%s\\total cost=%s, LF samples:%d; HF samples:%d' % (self.af_name,
                                                                               str(self.total_cost[-1]),
                                                                               len(self.model.X_l),
                                                                               len(self.model.X_h)))
                plt.savefig(self.dir + '%s_MACH_l_%d.png'
                            % (
                                self.af_name,
                                self.iteration,
                            ))

                # plt.show()
                plt.close()

                plt.plot(Y[:, 0], z_h_mesh[:, 19], label='HF simulation')
                plt.plot(Y[:, 0], self.mean_h.reshape(size, size).T[:, 17], label='MF model')
                plt.legend()
                plt.xlabel('MACH')
                plt.ylabel(r'$C_L$')
                plt.title('%s\\total cost=%s, LF samples:%d; HF samples:%d' % (self.af_name,
                                                                               str(self.total_cost[-1]),
                                                                               len(self.model.X_l),
                                                                               len(self.model.X_h)))
                plt.savefig(self.dir + '%s_MACH_h_%d.png'
                            % (
                                self.af_name,
                                self.iteration,
                            ))

                # plt.show()
                plt.close()

            if self.benchmark_name == 'NACA' and True:
                plt.plot(np.arange(1, 121), self.y_pool, label='True')
                plt.plot(np.arange(1, 121), self.mean_h, label='MF model')
                # plt.plot(np.arange(1, 121), self.mean_l, label='LF model')
                try:
                    plt.scatter(self.idx_h, self.model.Y_h, label='HF samples')
                    plt.scatter(self.idx_l, self.model.Y_l, label='LF samples')
                except:
                    pass
                plt.legend()
                # plt.show()
                plt.savefig(self.dir + '%s_%s_seed%d_%d_Model.png'
                            % (self.benchmark_name,
                               self.af_name,
                               self.seed,
                               self.iteration,
                               ))
                plt.close()
        return

    def testPerformance(self):
        y_opt = self.y_opt
        self.y_best = np.max(self.model.Y_h) if self.maximize else np.min(self.model.Y_h)

        best_i = np.argmax(self.mean_h) if self.maximize else np.argmin(self.mean_h)
        # f_x_best = self.f_h(self.x_pool[best_i])
        f_x_best = self.y_pool[best_i]

        self.inference_r.append(float(y_opt - f_x_best))
        self.simple_r.append(float(y_opt - self.y_best))

        self.rmse.append(float(RMSE(self.y_pool, self.mean_h)))
        # self.mae.append(float(MAE(self.y_pool, self.mean_h)))
        # self.mape.append(float(MAPE(self.y_pool, self.mean_h)))
        self.rsquare.append(float(Rsquare(self.y_pool, self.mean_h)))

        total_cost = len(self.model.Y_h) \
                     + len(self.model.Y_l) / self.cost
        self.total_cost.append(float(total_cost))

        return

    def saveResult(self):
        exp_para = '%s_seed%d' % (self.af_name, self.seed)
        path = self.dir + '%s_' % self.benchmark_name + exp_para

        np.savez(path,
                 X_l=self.model.X_l,
                 X_h=self.model.X_h,
                 SR=self.simple_r,
                 IR=self.inference_r,
                 RMSE=self.rmse,
                 # MAE=self.mae,
                 # MAPE=self.mape,
                 R_square=self.rsquare,
                 total_cost=self.total_cost)

        df_list = []
        sheet_list = []
        try:
            df = pd.DataFrame({'total_cost': self.total_cost,
                               'SR': self.simple_r,
                               'IR': self.inference_r,
                               'RMSE': self.rmse,
                               # 'MAE': self.mae,
                               # 'MAPE': self.mape,
                               'R_square': self.rsquare})
            df_list.append(df)
            sheet_list.append('Result')
        except:
            print('Failed to save result into excel for %s.' % exp_para)

        try:
            if self.input_dim == 1:
                df_LF = pd.DataFrame({'X_l0': self.model.X_l.ravel(),
                                      'Y_l': self.model.Y_l.ravel()})

            elif self.input_dim == 2:
                df_LF = pd.DataFrame({'X_l0': self.model.X_l[:, 0].ravel(),
                                      'X_l1': self.model.X_l[:, 1].ravel(),
                                      'Y_l': self.model.Y_l.ravel()})
            elif self.input_dim == 4:
                df_LF = pd.DataFrame({'X_l0': self.model.X_l[:, 0].ravel(),
                                      'X_l1': self.model.X_l[:, 1].ravel(),
                                      'X_l2': self.model.X_l[:, 2].ravel(),
                                      'X_l3': self.model.X_l[:, 3].ravel(),
                                      'Y_l': self.model.Y_l.ravel()})
            elif self.input_dim == 8:
                df_LF = pd.DataFrame({'X_l0': self.model.X_l[:, 0].ravel(),
                                      'X_l1': self.model.X_l[:, 1].ravel(),
                                      'X_l2': self.model.X_l[:, 2].ravel(),
                                      'X_l3': self.model.X_l[:, 3].ravel(),
                                      'X_l4': self.model.X_l[:, 4].ravel(),
                                      'X_l5': self.model.X_l[:, 5].ravel(),
                                      'X_l6': self.model.X_l[:, 6].ravel(),
                                      'X_l7': self.model.X_l[:, 7].ravel(),
                                      'Y_l': self.model.Y_l.ravel()})
            elif self.input_dim == 9:
                df_LF = pd.DataFrame({'X_l0': self.model.X_l[:, 0].ravel(),
                                      'X_l1': self.model.X_l[:, 1].ravel(),
                                      'X_l2': self.model.X_l[:, 2].ravel(),
                                      'X_l3': self.model.X_l[:, 3].ravel(),
                                      'X_l4': self.model.X_l[:, 4].ravel(),
                                      'X_l5': self.model.X_l[:, 5].ravel(),
                                      'X_l6': self.model.X_l[:, 6].ravel(),
                                      'X_l7': self.model.X_l[:, 7].ravel(),
                                      'X_l8': self.model.X_l[:, 8].ravel(),
                                      'Y_l': self.model.Y_l.ravel()})
            df_list.append(df_LF)
            sheet_list.append('LF_samples')
        except:
            print('Failed to save LF samples into excel for %s.' % exp_para)

        try:
            if self.input_dim == 1:
                df_HF = pd.DataFrame({'X_h0': self.model.X_h.ravel(),
                                      'Y_h': self.model.Y_h.ravel()})
            elif self.input_dim == 2:
                df_HF = pd.DataFrame({'X_h0': self.model.X_h[:, 0].ravel(),
                                      'X_h1': self.model.X_h[:, 1].ravel(),
                                      'Y_h': self.model.Y_h.ravel()})
            elif self.input_dim == 4:
                df_HF = pd.DataFrame({'X_h0': self.model.X_h[:, 0].ravel(),
                                      'X_h1': self.model.X_h[:, 1].ravel(),
                                      'X_h2': self.model.X_h[:, 2].ravel(),
                                      'X_h3': self.model.X_h[:, 3].ravel(),
                                      'Y_h': self.model.Y_h.ravel()})
            elif self.input_dim == 8:
                df_HF = pd.DataFrame({'X_h0': self.model.X_h[:, 0].ravel(),
                                      'X_h1': self.model.X_h[:, 1].ravel(),
                                      'X_h2': self.model.X_h[:, 2].ravel(),
                                      'X_h3': self.model.X_h[:, 3].ravel(),
                                      'X_h4': self.model.X_h[:, 4].ravel(),
                                      'X_h5': self.model.X_h[:, 5].ravel(),
                                      'X_h6': self.model.X_h[:, 6].ravel(),
                                      'X_h7': self.model.X_h[:, 7].ravel(),
                                      'Y_h': self.model.Y_h.ravel()})
            elif self.input_dim == 9:
                df_HF = pd.DataFrame({'X_h0': self.model.X_h[:, 0].ravel(),
                                      'X_h1': self.model.X_h[:, 1].ravel(),
                                      'X_h2': self.model.X_h[:, 2].ravel(),
                                      'X_h3': self.model.X_h[:, 3].ravel(),
                                      'X_h4': self.model.X_h[:, 4].ravel(),
                                      'X_h5': self.model.X_h[:, 5].ravel(),
                                      'X_h6': self.model.X_h[:, 6].ravel(),
                                      'X_h7': self.model.X_h[:, 7].ravel(),
                                      'X_h8': self.model.X_h[:, 8].ravel(),
                                      'Y_h': self.model.Y_h.ravel()})
            df_list.append(df_HF)
            sheet_list.append('HF_samples')
        except:
            print('Failed to save HF samples into excel for %s.' % exp_para)

        try:
            df = pd.DataFrame({'Idx_h': np.squeeze(self.idx_h), })
            df_list.append(df)
            sheet_list.append('Idx_h')
        except:
            print('Failed to save Idx_h into excel for %s.' % exp_para)

        try:
            df = pd.DataFrame({'Idx_l': np.squeeze(self.idx_l), })
            df_list.append(df)
            sheet_list.append('Idx_l')
        except:
            print('Failed to save Idx_l into excel for %s.' % exp_para)

        with pd.ExcelWriter(path + '.xlsx') as writer:
            for i in range(len(df_list)):
                df_list[i].to_excel(writer, sheet_name=sheet_list[i], index=1)

            # df.to_excel(writer, sheet_name=exp_para, index=1)
            # df_LF.to_excel(writer, sheet_name='LF_samples', index=1)
            # df_HF.to_excel(writer, sheet_name='HF_samples', index=1)
        return

    def dmes(self, n_sample: int = 100):
        # grid_size = 10
        # grid_1d = np.linspace(0, 1, grid_size)
        # x_pool = cartesian([grid_1d for _ in range(self.input_dim)])
        # mean_h = self.model.meanPredict(x_pool)
        # cov_h = self.model.varPredict(x_pool, full_cov=True)  # self.
        #
        # y_best_set = Opt_Random_Sample(mean_h,
        #                                cov_h,  # self.cov_h
        #                                Size=n_sample,
        #                                RNG=self.rng,
        #                                maximize=self.maximize)
        y_best_set = Opt_Random_Sample(model=self.model,
                                       input_dim=self.input_dim,
                                       Size=n_sample,
                                       RNG=self.rng,
                                       maximize=self.maximize)

        af_h, af_l = dmes_compute(y_best_set=y_best_set,  # self.y_best_set
                                  mean_h=self.mean_h,
                                  std_h=self.std_h,
                                  std_l=self.std_l,
                                  rho=self.model.rho,
                                  cost=self.cost,
                                  maximize=self.maximize)

        return af_h, af_l

    def vfei(self):
        af_h = ei_compute(mean=self.mean_h,
                          std=self.std_h,
                          y_best=self.y_best,
                          x0=0,
                          maximize=self.maximize)

        '''?'''
        # af_h = af_h / self.cost

        af_l = ei_compute(mean=self.mean_h,
                          std=abs(self.std_l * self.model.rho),
                          y_best=self.y_best,
                          x0=0,
                          maximize=self.maximize)
        return af_h, af_l

    # def random(self):
    #     self.next_i = self.rng.choice(np.setdiff1d(np.arange(self.x_pool.shape[0]), self.observed_idx_h))
    #     self.next_x = self.x_pool[self.next_i]
    #     return

    def efi(self):
        af_h, af_l = efi_compute(y_best=self.y_best,
                                 mean_h=self.mean_h,
                                 std_h=self.std_h,
                                 std_l=self.std_l,
                                 rho=self.model.rho,
                                 cost=self.cost,
                                 maximize=self.maximize)
        return af_h, af_l

    def ei(self):
        y_best_l = np.max(self.model.Y_l) if self.maximize else np.min(self.model.Y_l)

        af_h = ei_compute(mean=self.mean_h,
                          std=self.std_h,
                          y_best=self.y_best,
                          maximize=self.maximize)

        # af_l = ei_compute(mean=self.mean_l,
        #                   std=self.std_l,
        #                   y_best=y_best_l,
        #                   maximize=self.maximize)
        af_l = np.zeros(af_h.shape)
        return af_h, af_l

    def mes(self, n_sample: int = 100):
        # cov_h = self.model.varPredict(self.x_pool, full_cov=True)
        # y_best_set = Opt_Random_Sample(self.mean_h,
        #                                cov_h,
        #                                Size=n_sample,
        #                                RNG=self.rng,
        #                                maximize=self.maximize)

        # cov_l = self.model.PCKLf.varPredict(self.x_pool, full_cov=True)
        # y_best_set_l = Opt_Random_Sample(self.mean_l,
        #                                  cov_l,
        #                                  Size=n_sample,
        #                                  RNG=self.rng,
        #                                  maximize=self.maximize)

        y_best_set = Opt_Random_Sample(model=self.model,
                                       input_dim=self.input_dim,
                                       Size=n_sample,
                                       RNG=self.rng,
                                       maximize=self.maximize)

        af_h = mes_compute(y_best_set=y_best_set,
                           mean=self.mean_h,
                           std=self.std_h,
                           maximize=self.maximize)

        # y_best_set_l = Opt_Random_Sample(model=self.model.PCKLf,
        #                                  input_dim=self.input_dim,
        #                                  Size=n_sample,
        #                                  RNG=self.rng,
        #                                  maximize=self.maximize)

        # y_best_set_l = Opt_Random_Sample(model=self.model.modelLf,
        #                                  input_dim=self.input_dim,
        #                                  Size=n_sample,
        #                                  RNG=self.rng,
        #                                  maximize=self.maximize)
        # af_l = mes_compute(y_best_set=y_best_set_l,
        #                    mean=self.mean_l,
        #                    std=self.std_l,
        #                    maximize=self.maximize)
        af_l = np.zeros(af_h.shape)
        return af_h, af_l

    def vfmes(self, n_sample: int = 100):
        # cov_h = self.model.varPredict(self.x_pool, full_cov=True)
        # y_best_set = Opt_Random_Sample(self.mean_h,
        #                                cov_h,
        #                                Size=n_sample,
        #                                RNG=self.rng,
        #                                maximize=self.maximize)

        # cov_l = self.model.PCKLf.varPredict(self.x_pool, full_cov=True)
        # y_best_set_l = Opt_Random_Sample(self.mean_l,
        #                                  cov_l,
        #                                  Size=n_sample,
        #                                  RNG=self.rng,
        #                                  maximize=self.maximize)

        y_best_set = Opt_Random_Sample(model=self.model,
                                       input_dim=self.input_dim,
                                       Size=n_sample,
                                       RNG=self.rng,
                                       maximize=self.maximize)

        af_h = mes_compute(y_best_set=y_best_set,
                           mean=self.mean_h,
                           std=self.std_h,
                           maximize=self.maximize)
        af_h = af_h / self.cost

        af_l = mes_compute(y_best_set=y_best_set,
                           mean=self.mean_h,
                           std=abs(self.std_l * self.model.rho),
                           maximize=self.maximize)

        return af_h, af_l
