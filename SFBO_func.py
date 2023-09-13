from BO_compute_func import Opt_Random_Sample, mes_compute
from Test_func import RMSE, IR, SR, Rsquare
import functools as ft
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Kriging_func import Kriging

class SFBO():
    def __init__(self,
                 result_dir,
                 input_dim,
                 benchmark_name,
                 x_pool,
                 y_pool,
                 idx_h,
                 x_train_h,
                 y_train_h,
                 af_name: str,
                 seed: int = 0,
                 cost: int = 10,
                 maximize: bool = True):
        self.result_dir = result_dir
        self.input_dim = input_dim
        self.benchmark_name = benchmark_name
        self.x_pool = x_pool
        self.y_pool = y_pool
        self.idx_h = idx_h
        self.af_name = af_name
        self.seed = seed
        self.cost = cost
        self.maximize = maximize

        self.model = Kriging(x_train_h,
                             y_train_h,
                             input_dim,
                             Rtype=11)
        self.model.fitModel()

        self.simple_r = []
        self.inference_r = []
        self.rmse = []
        self.mae = []
        self.mape = []
        self.rsquare = []
        self.total_cost = []
        self.iteration = 0
        self.preCompute()

    def nextObservation(self):
        af = self.mes()

        i = -1
        while abs(i) < len(af) and any(np.allclose(self.x_pool[np.argsort(af)[i]], self.model.X[j]) for j in
                                       range(len(self.model.X))):
            i = i - 1
        max_i_h = np.argsort(af)[i]

        max_i = max_i_h
        next_x = self.x_pool[max_i]
        next_x_af_value = af[max_i]

        next_y = self.y_pool[max_i]
        self.idx_h = np.vstack((self.idx_h, max_i))
        if True:
            plt.plot(af, label='HF')

            plt.legend()
            plt.savefig(self.result_dir + '%s_%s_seed%d_%d_S.png'
                        % (self.benchmark_name,
                           self.af_name,
                           self.seed,
                           self.iteration))
            # plt.show()
            plt.close()

        '''3.  Update the model. '''
        self.model.addSample(next_x,
                        next_y)

        '''4. Ready for next observation.'''
        self.iteration += 1
        self.preCompute()

        return


    def preCompute(self, plot_performance=True):
        self.mean = self.model.meanPredict(self.x_pool)
        self.std = self.model.stdPredict(self.x_pool)

        ''' Test the model performance.'''
        self.testPerformance()
        self.saveResult()
        if plot_performance:
            plt.plot(np.arange(1, 121), self.y_pool, label='True')
            plt.plot(np.arange(1, 121), self.mean, label='Model')

            try:
                plt.scatter(self.idx_h, self.model.Y, label='HF samples')
            except:
                pass
            plt.legend()
            # plt.show()
            plt.savefig(self.result_dir + '%s_%s_seed%d_%d_Model.png'
                        % (self.benchmark_name,
                           self.af_name,
                           self.seed,
                           self.iteration,
                           ))
            plt.close()
        return


    def testPerformance(self, ):
        y_opt = np.max(self.y_pool) if self.maximize else np.min(self.y_pool)
        y_best = np.max(self.model.Y) if self.maximize else np.min(self.model.Y)

        best_i = np.argmax(self.mean) if self.maximize else np.argmin(self.mean)
        # f_x_best = f_h(x_pool[best_i])
        f_x_best = self.y_pool[best_i]

        self.inference_r.append(float(y_opt - f_x_best))
        self.simple_r.append(float(y_opt - y_best))

        self.rmse.append(float(RMSE(self.y_pool, self.mean)))
        # mae.append(float(MAE(y_pool, mean)))
        # mape.append(float(MAPE(y_pool, mean)))
        self.rsquare.append(float(Rsquare(self.y_pool, self.mean)))

        total_cost = len(self.model.Y)
        self.total_cost.append(float(total_cost))

        return


    def saveResult(self):
        exp_para = '%s_seed%d' % (self.af_name, self.seed)
        path = self.result_dir + '%s_' % self.benchmark_name + exp_para

        np.savez(path,
                 X_h=self.model.X,
                 SR=self.simple_r,
                 IR=self.inference_r,
                 RMSE=self.rmse,
                 # MAE=mae,
                 # MAPE=mape,
                 R_square=self.rsquare,
                 total_cost=self.total_cost)

        df_list = []
        sheet_list = []
        try:
            df = pd.DataFrame({'total_cost': self.total_cost,
                               'SR': self.simple_r,
                               'IR': self.inference_r,
                               'RMSE': self.rmse,
                               # 'MAE': mae,
                               # 'MAPE': mape,
                               'R_square': self.rsquare})
            df_list.append(df)
            sheet_list.append('Result')
        except:
            print('Failed to save result into excel for %s.' % exp_para)

        try:
            df_HF = pd.DataFrame({'X_h0': self.model.X[:, 0].ravel(),
                                  'X_h1': self.model.X[:, 1].ravel(),
                                  'X_h2': self.model.X[:, 2].ravel(),
                                  'X_h3': self.model.X[:, 3].ravel(),
                                  'X_h4': self.model.X[:, 4].ravel(),
                                  'X_h5': self.model.X[:, 5].ravel(),
                                  'X_h6': self.model.X[:, 6].ravel(),
                                  'X_h7': self.model.X[:, 7].ravel(),
                                  'X_h8': self.model.X[:, 8].ravel(),
                                  'Y_h': self.model.Y.ravel()})
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

        with pd.ExcelWriter(path + '.xlsx') as writer:
            for i in range(len(df_list)):
                df_list[i].to_excel(writer, sheet_name=sheet_list[i], index=1)

            # df.to_excel(writer, sheet_name=exp_para, index=1)
            # df_LF.to_excel(writer, sheet_name='LF_samples', index=1)
            # df_HF.to_excel(writer, sheet_name='HF_samples', index=1)
        return


    def mes(self, ):
        y_best_set = Opt_Random_Sample(model=self.model,
                                       input_dim=9,
                                       Size=100,
                                       RNG=np.random.default_rng(0),
                                       maximize=self.maximize)

        af = mes_compute(y_best_set=y_best_set,
                         mean=self.mean,
                         std=self.std,
                         maximize=self.maximize)
        return af






