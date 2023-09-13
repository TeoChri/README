import itertools as it
import numpy as np
from joblib import delayed, Parallel
from BO_func import MF_BO as BO
from BO_GA_func import MF_BO as BO_GA
from plotResult_func import plotResult
from benchmark_func import *
from pyDOE import lhs
import pandas as pd


# from mfgpr import GaussianProcessCoKriging as MF
# from CoKriging_func import MultiFidelityModel as MF
# from MF_func import MultiFidelityModel as MF

def main():
    '''1. Initial setup.'''
    af_name_list = [
                    'DMES',
                    'VFEI',
                    'EFI',
                    'MES', ]

    '''2. Initial DoE.'''
    if bench_name == 'RAE2822':
        grid_size = 20
        maximize = 1

        x_mesh = pd.read_csv('C:\\Users\\Teo\\Desktop\\1\\grid_size%d\\exp_HF.csv' % grid_size, skipinitialspace=True)
        x_mesh.columns = x_mesh.columns.str.strip()
        x_pool = x_mesh[["AOA", "MACH_NUMBER"]].values

        # x_0 = pd.read_csv('C:\\Users\\Teo\\Desktop\\1\\grid_size%d\\exp_HF_0.csv'%grid_size, skipinitialspace=True)
        # x_0.columns = x_0.columns.str.strip()
        # x_0 = x_0['0'].values
        #
        # x_1 = pd.read_csv('C:\\Users\\Teo\\Desktop\\1\\grid_size%d\\exp_HF_1.csv'%grid_size, skipinitialspace=True)
        # x_1.columns = x_1.columns.str.strip()
        # x_1 = x_1['0'].values
        #
        # lb = np.array([[x_0[0], x_1[0]]])
        # ub = np.array([[x_0[-1], x_1[-1]]])
        #
        # xTrain_l = lhs(n=input_dim, samples=n_init_l) * (ub-lb) + lb
        # xTrain_h = lhs(n=input_dim, samples=n_init_h) * (ub - lb) + lb
        # try :
        #     # xTrain_l= pd.read_excel(path+'2_%d_RAE2822_DMES_seed0.xlsx'%iter, sheet_name='DMES_seed0_LF_samples')[
        #     #                ['DMES_seed0_xTrainLF0', 'DMES_seed0_xTrainLF1']].values[
        #     #            :n_init_l, ::]
        #     #
        #     # xTrain_h = pd.read_excel(path+'2_%d_RAE2822_DMES_seed0.xlsx'%iter, sheet_name='DMES_seed0_HF_samples')[
        #     #                ['DMES_seed0_xTrainHF0', 'DMES_seed0_xTrainHF1']].values[
        #     #            :n_init_h, ::]
        #     pass
        #
        # except:
        idx_l = (lhs(n=input_dim, samples=n_init_l) * grid_size).astype(int)
        idx_l = idx_l[:, 0] * grid_size + idx_l[:, 1]
        xTrain_l = x_pool[idx_l]

        idx_h = (lhs(n=input_dim, samples=n_init_h) * grid_size).astype(int)
        idx_h = idx_h[:, 0] * grid_size + idx_h[:, 1]
        xTrain_h = x_pool[idx_h]
    elif bench_name == 'NACA':
        x_pool = pd.read_excel('C:\\Users\\Teo\\Desktop\\0504\\NACA0012\\m6_120samples_9input.xlsx').values

        idx_l = (lhs(n=1, samples=n_init_l) * 120).astype(int)
        xTrain_l = x_pool[idx_l]

        idx_h = (lhs(n=1, samples=n_init_h) * 120).astype(int)
        xTrain_h = x_pool[idx_h]

        grid_size = [idx_l, idx_h]
        maximize = 1
    else:
        grid_size = 100 if input_dim == 1 else 10
        maximize = 0

        from sklearn.utils.extmath import cartesian
        grid_1d = np.linspace(0, 1, grid_size)
        x_pool = cartesian([grid_1d for _ in range(input_dim)])

        xTrain_l = lhs(n=input_dim,
                       samples=n_init_l)
        idx = np.random.choice(xTrain_l.shape[0],
                               size=n_init_h,
                               replace=False,
                               p=None)
        xTrain_h = xTrain_l[idx]

    def one_exp(af_name, cost):  # bench_name,
        """
        One experiment.
        """
        result_dir = path + '%d_%d_' % (cost, iter)

        '''3. Fit the model.
              Then, Bayesian Optimization.'''
        if input_dim <= 2 :
            bo = BO(dir=result_dir,
                    input_dim=input_dim,
                    benchmark_name=bench_name,
                    grid_size=grid_size,
                    x_pool=x_pool,
                    x_train_l=xTrain_l,
                    x_train_h=xTrain_h,
                    af_name=af_name,
                    seed=0,
                    cost=cost,
                    maximize=maximize)

            while bo.total_cost[-1] < total_cost:
                bo.next_observation(plot_perfomance=True,
                                    plot_af=True)
        elif bench_name=='NACA':
            bo = BO(dir=result_dir,
                    input_dim=input_dim,
                    benchmark_name=bench_name,
                    grid_size=grid_size,
                    x_pool=x_pool,
                    x_train_l=xTrain_l,
                    x_train_h=xTrain_h,
                    af_name=af_name,
                    seed=0,
                    cost=cost,
                    maximize=maximize)

            while bo.total_cost[-1] < total_cost:
                bo.next_observation(plot_perfomance=True,
                                    plot_af=1)
        else:
            bo = BO_GA(dir=result_dir,
                       input_dim=input_dim,
                       benchmark_name=bench_name,
                       grid_size=grid_size,
                       x_pool=x_pool,
                       x_train_l=xTrain_l,
                       x_train_h=xTrain_h,
                       af_name=af_name,
                       seed=0,
                       cost=cost,
                       maximize=maximize)
            while bo.total_cost[-1] < total_cost:
                lb = np.tile(np.array([0]), input_dim)
                ub = np.tile(np.array([1]), input_dim)

                bo.next_observation(lb=lb,
                                    ub=ub,
                                    size_pop=100,
                                    iter_pop=50,
                                    plot_af=0)

        # Save the result.
        bo.saveResult()

        measure_list = [
                        'SR',
                        'IR',
                        'RMSE',
                        'R_square',
                        ]
        plotResult(path=result_dir,
                   bench_name_list=[bench_name],
                   af_name_list=af_name_list,
                   seed_list=[0],
                   measure_list=measure_list)

    '''4. Repeat the experiments with the experiment parameters.'''
    Parallel(n_jobs=parall_num, verbose=10)([delayed(one_exp)(AF, COST)
                                             for AF, COST in
                                             it.product(af_name_list, cost_list)])

    '''5. Compare the experiment result with different measurement.
          And plot them.'''


if __name__ == '__main__':
    cost_list = [2, 5, 10]
    path = 'C:\\Users\\Teo\\Desktop\\0504\\NACA03\\'

    # path = 'C:\\Users\\Teo\\Desktop\\0504\\RAE05\\'
    bench_name_list = [
        # dim = 1
        # 'Forrester 1a',
        # 'Forrester 1b',
        # 'Forrester 1c',
        # 'Sasena',
        # 'Gramacy-Lee',
        # dim=2
        # 'Currin',
        # 'Bukin',
        # 'Branin',
        # 'Six-hump camel-back',
        # 'Booth',
        # 'Bohachevsky',
        # dim=4
        # 'Park',
        # 'Park2',
        # dim=8
        # 'Borehole',
        # 'RAE2822',
        'NACA'
    ]

    # bench_name = 'Forrester 1a'
    # input_dim = 1
    # n_init_h = 3
    # n_init_l = 5
    # total_cost = 15

    # # bench_name = 'Currin' #'RAE2822'
    # input_dim = 2
    # n_init_h = 5
    # n_init_l = 15
    # total_cost = 50

    # input_dim = 4
    # n_init_h = 5
    # n_init_l = 20
    # total_cost = 100

    input_dim = 9
    n_init_h = 3
    n_init_l = 20
    total_cost = 50

    parall_num = -2
    for bench_name in bench_name_list:
        for iter in range(5):
            main()
