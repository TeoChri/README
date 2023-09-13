import copy
from pyDOE import lhs
from sklearn.utils.extmath import cartesian
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from benchmark_func import *
from Test_func import RMSE, MAE, Rsquare, MAPE
from CoKriging_func import CoKriging as MF
from BO_compute_func import Opt_Random_Sample, mes_compute, dmes_compute, ei_compute, efi_compute


def main():

    '''1. Initial setup.'''
# # Set up the path for saving result later.
# result_dir = 'C:\\Users\\Teo\\Desktop\\BO Result09\\10_'
    bench_name = 'Forrester 1a'
    input_dim = 1
    # n_init_h = 3
    n_init_l = 5


    grid_size = 100
    maximize = 0



    '''2. Initial DoE.'''
    # xTrain_l = np.linspace(0,1, 5)
    # xTrain_h = np.linspace(0, 1, 3)

    # data = np.load('C:\\Users\\Teo\\Desktop\\test_14cost20\\0_result.npz')
    # xTrain_l = data['X_l'][:5]
    # xTrain_h = data['X_h'][:3]
    try:
        data = np.load(path+'cost5_0_00.npz')
        xTrain_l = data['X_l'][:n_init_l]
    except:
        xTrain_l = lhs(n=input_dim,
                       samples=n_init_l)

    # xTrain_h = lhs(n=input_dim,
    #                samples=n_init_h)
    # idx = np.random.choice(xTrain_l.shape[0],
    #                        size=n_init_h,
    #                        replace=False,
    #                        p=None)
    # idx = np.array([2,3,4])

    idx = np.array([[0,1,2],
                    [0,1,3],
                    [0,1,4],
                    [1,2,3],
                    [1,2,4],
                    [2,3,4]])

    for i in range(len(idx)):
        result_dir = path + 'cost%d_%d_'%(cost_h, i)

        xTrain_h = xTrain_l[idx[i,:]]
        def one_exp(bench_name, seed):
            """
            One experiment.
            """

            '''3. Fit the model.
                  Then, Bayesian Optimization.'''
            f_h_compute = {'Forrester 1a': FO,
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
            assert bench_name in f_h_compute.keys(), f'Unknown benchmark function: {bench_name}'
            f_l_compute = {'Forrester 1a': FOa_low,
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
            assert bench_name in f_l_compute.keys(), f'Unknown benchmark function: {bench_name}'

            f_h = f_h_compute[bench_name]
            f_l = f_l_compute[bench_name]

            '''2. Initialize the MF model.'''
            grid_1d = np.linspace(0, 1, grid_size)
            x_pool = cartesian([grid_1d for _ in range(input_dim)])
            y_pool_l = f_l(x_pool)
            y_pool_h = f_h(x_pool)

            y_train_h = f_h(xTrain_h)
            y_train_l = f_l(xTrain_l)
            model = MF(xTrain_l, y_train_l, xTrain_h, y_train_h,dim=input_dim)
            model.fitModel()

            model1 = copy.deepcopy(model)
            model2 = copy.deepcopy(model)

            mean_h = model.meanPredict(x_pool)
            mean_l = model.meanLowPredict(x_pool)
            std_h = model.stdPredict(x_pool)
            std_l = model.stdLowPredict(x_pool)
            rho = model.rho

            def result_compute(result, model, mean):
                y_opt = np.max(y_pool_h) if maximize else np.min(y_pool_h)
                y_best = np.max(model.Y_h) if maximize else np.min(model.Y_h)

                best_i = np.argmax(mean) if maximize else np.argmin(mean)
                f_x_best = f_h(x_pool[best_i])

                result['IR'].append(float(y_opt - f_x_best))
                result['SR'].append(float(y_opt - y_best))
                result['RMSE'].append(float(RMSE(y_pool_h, mean)))
                result['R_square'].append(float(Rsquare(y_pool_h, mean)))

                total_cost = len(model.Y_h) \
                             + len(model.Y_l)/cost_h
                result['total_cost'].append(float(total_cost))
                return result

            def stop_criterion(result, scale=0.05):
                measure = 'RMSE'
                value = result[measure]
                if len(value) < 2:
                    res = True
                else:
                    # res = abs(value[-1] - value[-2]) > (np.max(value) * scale)
                    res = abs(value[-1]) > (np.max(value) * scale)

                if res:
                    res = result['total_cost'][-1] < 15
                return res

            result = {'RMSE': [],
                      'R_square': [],
                      'SR': [],
                      'IR': [],
                      'total_cost': [],
                      }
            result = result_compute(result, model, mean_h)
            # plot the model
            if True:
                plt.figure()
                plt.scatter(model.X_h,
                            model.Y_h,
                            label='HF samples : %d' % len(model.X_h),
                            marker='o')  # , label='HF sampled points'
                plt.scatter(model.X_l,
                            model.Y_l,
                            label='LF samples : %d' % len(model.X_l),
                            marker='*')  # , label='LF sampled points'

                # from Kriging_func import Kriging
                # modelh = Kriging(xTrain_h, y_train_h,1)
                # plt.plot(x_pool, modelh.meanPredict(x_pool), label='K_h')
                # plt.plot(x_pool, modelh.meanPredict(x_pool)+modelh.varPredict(x_pool), label='up')
                # plt.plot(x_pool, modelh.meanPredict(x_pool)-modelh.varPredict(x_pool), label='lo')
                plt.plot(x_pool,
                         y_pool_h,
                         label='True HF model')
                plt.plot(x_pool,
                         y_pool_l,
                         label='True LF model')

                plt.plot(x_pool,
                         mean_h,
                         linestyle='-.',
                         # label='MF predict',
                         )
                plt.fill_between(np.squeeze(x_pool),
                                 np.squeeze(mean_h + 3 * std_h),
                                 np.squeeze(mean_h - 3 * std_h),
                                 alpha=0.3,
                                 label='MF model confidence')

                plt.plot(x_pool,
                         mean_l,
                         linestyle=':',
                         # label='LF model predict',
                         )
                plt.fill_between(np.squeeze(x_pool),
                                 np.squeeze(mean_l + 3 * std_l),
                                 np.squeeze(mean_l - 3 * std_l),
                                 alpha=0.3,
                                 label='LF model confidence')
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.title('%s,  \n model, total_cost=%s'  # seed=%d
                          % (bench_name, str(result['total_cost'][-1])))
                plt.legend()
                plt.savefig(result_dir + 'Model.png')
                plt.show()
                plt.close()
            # opt and plot
            if True:
                y_best_set = Opt_Random_Sample(model=model,
                                               input_dim=input_dim,
                                               Size=100,
                                               RNG=np.random.default_rng(seed),
                                               maximize=maximize)

                # f(HF)
                mes_h = mes_compute(y_best_set,
                                    mean=mean_h,
                                    std=std_h,
                                    maximize=maximize)
                # res_h = mes_h / cost  # /3

                # f(LF)
                mes_l = mes_compute(y_best_set,
                                    mean=mean_h,
                                    std=abs(std_l * rho),
                                    maximize=maximize)

                # f(HF) - f(HF-LF)
                var = std_h ** 2 - rho ** 2 * std_l ** 2
                std = np.zeros((len(var), 1))
                for j in range(len(var)):
                    std[j] = np.sqrt(var[j]) if var[j] > 0 else std_h[j]
                mes_h_on_l = mes_compute(y_best_set,
                                         mean=mean_h,
                                         std=std,
                                         # cov_l_h,
                                         maximize=maximize, )
                mes_h_l = mes_h - mes_h_on_l

                plt.figure()
                plt.plot(x_pool, mes_h, label=r'$MES_2$')
                plt.plot(x_pool, mes_l, label=r'$MES_{2\_1}$')
                plt.plot(x_pool, mes_h_l, label=r'$MES_2-MES_{2\backslash1}$')
                plt.legend()
                plt.savefig(result_dir + 'AF.png')
                plt.show()
                plt.close()
            # plot the variance
            if True:
                plt.figure()
                plt.plot(x_pool,
                         std_h ** 2,
                         label=r'$v_2$')
                plt.plot(x_pool,
                         rho ** 2 * std_l ** 2,
                         label=r'$v_{2\_1}$')
                plt.plot(x_pool,
                         var,
                         label=r'$v_{2\backslash1}$')
                plt.xlabel('X')
                # plt.ylabel('Y')
                plt.title('%s,  \n Variance'  # seed=%d
                          % bench_name)

                plt.legend()
                plt.savefig(result_dir + 'Variance.png')
                plt.show()
                plt.close()
            # save the data
            if True:
                np.savez(result_dir+'00',
                         X_l=xTrain_l,
                         X_h=xTrain_l,
                         y_train_l=y_train_l,
                         y_train_h=y_train_h,

                         # hyp_l=model.modelLf.kriging.hyperParams,
                         AF_h=mes_h,
                         AF_l=mes_l,
                         AF_h_l=mes_h_l,
                         RMSE=result['RMSE'],
                         R_square=result['R_square'],
                         IR=result['IR'],
                         SR=result['SR'],
                         total_cost=result['total_cost'],
                         )

                df_list = []
                sheet_list = []
                try:
                    df_result = pd.DataFrame({'RMSE': result['RMSE'],
                                              'R_square': result['R_square'],
                                              'IR': result['IR'],
                                              'SR': result['SR'],
                                              'total_cost': result['total_cost'],
                                              })
                    df_list.append(df_result)
                    sheet_list.append('_result')
                except:
                    print('Failed to save result into excel.')
                try:
                    df_train_h = pd.DataFrame({'xTrain_l': xTrain_l.ravel(),
                                               'yTrain_l': y_train_l.ravel(), })
                    df_list.append(df_train_h)
                    sheet_list.append('LF Train')
                except:
                    print('Failed to save LF Train into excel.')
                try:
                    df_train_h = pd.DataFrame({'xTrain_h': xTrain_h.ravel(),
                                               'yTrain_h': y_train_h.ravel(), })
                    df_list.append(df_train_h)
                    sheet_list.append('HF Train')
                except:
                    print('Failed to save HF Train into excel.')


                try:
                    df_af = pd.DataFrame({'AF_h': mes_h.ravel(),
                                          'AF_l': mes_l.ravel(),
                                          'AF_h_l': mes_h_l.ravel(), })
                    df_list.append(df_af)
                    sheet_list.append('_AF')
                except:
                    print('Failed to save AF into excel.')

                with pd.ExcelWriter(result_dir + '00.xlsx') as writer:
                    for i in range(len(df_list)):
                        df_list[i].to_excel(writer, sheet_name=sheet_list[i], index=1)

            result1 = copy.deepcopy(result)
            result2 = copy.deepcopy(result)
            mes_h1 = copy.deepcopy(mes_h)
            mes_h2 = copy.deepcopy(mes_h)

            # while iter < 10:
            iter = 0
            while stop_criterion(result1) and iter<20:
                iter += 1
                # update the model
                idx1 = np.argmax(mes_h1)
                idx2 = np.argmax(mes_l)
                if (mes_h1[idx1]/cost_h) <= mes_l[idx2]:
                    xnew1 = x_pool[idx2]
                    ynew1 = f_l(xnew1)
                    model1.addSample(xnew1, ynew1, 'LF')
                else:
                    xnew1 = x_pool[idx1]
                    ynew1 = f_h(xnew1)
                    model1.addSample(xnew1, ynew1, 'HF')
                # update the data
                mean_h1 = model1.meanPredict(x_pool)
                mean_l1 = model1.meanLowPredict(x_pool)
                std_h1 = model1.stdPredict(x_pool)
                std_l1 = model1.stdLowPredict(x_pool)
                rho1 = model1.rho
                # compute the result
                result1 = result_compute(result1, model1, mean_h1)
                # plot model1
                if True:
                    plt.figure()
                    plt.scatter(model1.X_h,
                                model1.Y_h,
                                label='HF samples : %d' % len(model1.X_h),
                                marker='o')  # , label='HF sampled points'
                    plt.scatter(model1.X_l,
                                model1.Y_l,
                                label='LF samples : %d' % len(model1.X_l),
                                marker='*')  # , label='LF sampled points'

                    plt.plot(x_pool,
                             f_h(x_pool),
                             label='True HF model')
                    plt.plot(x_pool,
                             f_l(x_pool),
                             label='True LF model')
                    plt.plot(x_pool,
                             mean_h1,
                             linestyle='-.',
                             # label='MF predict',
                             )
                    plt.fill_between(np.squeeze(x_pool),
                                     np.squeeze(mean_h1 + 3 * std_h1),
                                     np.squeeze(mean_h1 - 3 * std_h1),
                                     alpha=0.3,
                                     label='MF model confidence')
                    plt.plot(x_pool,
                             mean_l1,
                             linestyle=':',
                             # label='LF model predict',
                             )
                    plt.fill_between(np.squeeze(x_pool),
                                     np.squeeze(mean_l1 + 3 * std_l1),
                                     np.squeeze(mean_l1 - 3 * std_l1),
                                     alpha=0.3,
                                     label='LF model confidence')
                    plt.scatter(xnew1,
                                ynew1,
                                label='new sample',
                                marker='^', linewidths=1, edgecolors='r')  # , label='new sampled point'
                    plt.xlabel('X')
                    plt.ylabel('Y')
                    plt.title('%s,  \n model, total_cost=%s'  # seed=%d
                              % (bench_name, str(result1['total_cost'][-1])))
                    plt.legend()
                    plt.savefig(result_dir + 'Model1_%d.png' % iter)
                    plt.show()
                    plt.close()
                if True:
                    plt.figure()
                    plt.plot(x_pool, mes_h1, label=r'$MES_2$')
                    plt.plot(x_pool, mes_l, label=r'$MES_{2\_1}$')
                    # plt.plot(x_pool, mes_h_l, label=r'$MES_{2\backslash1}$')
                    plt.legend()
                    plt.savefig(result_dir + 'AF1_%d.png'%iter)
                    plt.show()
                    plt.close()
                    # plot the variance
                if True:
                    plt.figure()
                    plt.plot(x_pool,
                             std_h1 ** 2,
                             label=r'$v_2$')
                    plt.plot(x_pool,
                             rho1 ** 2 * std_l1 ** 2,
                             label=r'$v_{2\_1}$')
                    # plt.plot(x_pool,
                    #          var,
                    #          label=r'$v_{2\backslash1}$')
                    plt.xlabel('X')
                    # plt.ylabel('Y')
                    plt.title('%s,  \n Variance'  # seed=%d
                              % bench_name)

                    plt.legend()
                    plt.savefig(result_dir + 'Variance1_%d.png'%iter)
                    plt.show()
                    plt.close()
                # save data1
                if True:
                    np.savez(result_dir + '1',
                             X_l=model1.X_l,
                             X_h=model1.X_h,
                             y_train_l=model1.Y_l,
                             y_train_h=model1.Y_h,

                             # hyp_l=model.modelLf.kriging.hyperParams,
                             AF_h=mes_h1,
                             AF_l=mes_l,
                             RMSE=result1['RMSE'],
                             R_square=result1['RMSE'],
                             IR=result1['IR'],
                             SR=result1['SR'],
                             total_cost=result1['total_cost'],
                             # AF_h_l=mes_h_l,
                             )

                    df_list = []
                    sheet_list = []
                    try:
                        df_result = pd.DataFrame({'RMSE': result1['RMSE'],
                                                  'R_square': result1['R_square'],
                                                  'IR': result1['IR'],
                                                  'SR': result1['SR'],
                                                  'total_cost': result1['total_cost'],
                                                  })
                        df_list.append(df_result)
                        sheet_list.append('_result')
                    except:
                        print('Failed to save result into excel.')
                    try:
                        df_train_h = pd.DataFrame({'xTrain_l': model1.X_l.ravel(),
                                                   'yTrain_l': model1.Y_l.ravel(), })
                        df_list.append(df_train_h)
                        sheet_list.append('LF Train')
                    except:
                        print('Failed to save LF Train into excel.')
                    try:
                        df_train_h = pd.DataFrame({'xTrain_h': model1.X_h.ravel(),
                                                   'yTrain_h': model1.Y_h.ravel(), })
                        df_list.append(df_train_h)
                        sheet_list.append('HF Train')
                    except:
                        print('Failed to save HF Train into excel.')

                    try:
                        df_af = pd.DataFrame({'AF_h': mes_h1.ravel(),
                                              'AF_l': mes_l.ravel(),
                                              # 'AF_h_l': mes_h_l.ravel(),
                                              })
                        df_list.append(df_af)
                        sheet_list.append('_AF')
                    except:
                        print('Failed to save AF into excel.')


                    with pd.ExcelWriter(result_dir + '1.xlsx') as writer:
                        for i in range(len(df_list)):
                            df_list[i].to_excel(writer, sheet_name=sheet_list[i], index=1)
                # pre opt model1
                if True:
                    y_best_set1 = Opt_Random_Sample(model=model1,
                                                    input_dim=input_dim,
                                                    Size=100,
                                                    RNG=np.random.default_rng(seed),
                                                    maximize=maximize)

                    # f(HF)
                    mes_h1 = mes_compute(y_best_set1,
                                         mean=mean_h1,
                                         std=std_h1,
                                         maximize=maximize)
                    # res_h = mes_h / cost  # /3

                    # f(LF)
                    mes_l = mes_compute(y_best_set1,
                                        mean=mean_h1,
                                        std=abs(std_l1 * rho1),
                                        maximize=maximize)

                if np.max(mes_h1)<=0 and np.max(mes_l)<=0:
                    break

            iter = 0
            while stop_criterion(result2) and iter<20:
                iter += 1
                # update the model
                idx3 = np.argmax(mes_h2)
                idx4 = np.argmax(mes_h_l)
                if (mes_h2[idx3] / cost_h) <= mes_h_l[idx4]:
                    xnew2 = x_pool[idx4]
                    ynew2 = f_l(xnew2)
                    model2.addSample(xnew2, ynew2, 'LF')
                else:
                    xnew2 = x_pool[idx3]
                    ynew2 = f_h(xnew2)
                    model2.addSample(xnew2, ynew2, 'HF')
                # update the data
                mean_h2 = model2.meanPredict(x_pool)
                mean_l2 = model2.meanLowPredict(x_pool)
                std_h2 = model2.stdPredict(x_pool)
                std_l2 = model2.stdLowPredict(x_pool)
                rho2 = model2.rho
                # compute the result
                result2 = result_compute(result2, model2, mean_h2)
                # plot model2
                if True:
                    plt.figure()
                    plt.scatter(model2.X_h,
                                model2.Y_h,
                                label='HF samples : %d' % len(model2.X_h),
                                marker='o')  # , label='HF sampled points'
                    plt.scatter(model2.X_l,
                                model2.Y_l,
                                label='LF samples : %d' % len(model2.X_l),
                                marker='*')  # , label='LF sampled points'
                    plt.plot(x_pool,
                             f_h(x_pool),
                             label='True HF model')
                    plt.plot(x_pool,
                             f_l(x_pool),
                             label='True LF model')

                    plt.plot(x_pool,
                             mean_h2,
                             linestyle='-.',
                             # label='MF predict',
                             )
                    plt.fill_between(np.squeeze(x_pool),
                                     np.squeeze(mean_h2 + 3 * std_h2),
                                     np.squeeze(mean_h2 - 3 * std_h2),
                                     alpha=0.3,
                                     label='MF model confidence')

                    plt.plot(x_pool,
                             mean_l2,
                             linestyle=':',
                             # label='LF model predict',
                             )
                    plt.fill_between(np.squeeze(x_pool),
                                     np.squeeze(mean_l2 + 3 * std_l2),
                                     np.squeeze(mean_l2 - 3 * std_l2),
                                     alpha=0.3,
                                     label='LF model confidence')
                    plt.scatter(xnew2,
                                ynew2,
                                label='new sample',
                                marker='^', linewidths=1, edgecolors='r')  # , label='new sampled point'
                    plt.xlabel('X')
                    plt.ylabel('Y')
                    plt.title('%s,  \n model, total_cost=%s'  # seed=%d
                              % (bench_name, str(result2['total_cost'][-1])))
                    plt.legend()
                    plt.savefig(result_dir + 'Model2_%d.png' % iter)
                    plt.show()
                    plt.close()
                if True:
                    plt.figure()
                    plt.plot(x_pool, mes_h2, label=r'$MES_2$')
                    # plt.plot(x_pool, mes_l, label=r'$MES_{2\_1}$')
                    plt.plot(x_pool, mes_h_l, label=r'$MES_{2\backslash1}$')
                    plt.legend()
                    plt.savefig(result_dir + 'AF2_%d.png'%iter)
                    plt.show()
                    plt.close()
                    # plot the variance
                if True:
                    plt.figure()
                    plt.plot(x_pool,
                             std_h2 ** 2,
                             label=r'$v_2$')
                    # plt.plot(x_pool,
                    #          rho ** 2 * std_l ** 2,
                    #          label=r'$v_{2\_1}$')
                    plt.plot(x_pool,
                             rho2 ** 2 * std_l2 ** 2,
                             label=r'$v_{2\backslash1}$')
                    plt.xlabel('X')
                    # plt.ylabel('Y')
                    plt.title('%s,  \n Variance'  # seed=%d
                              % bench_name)

                    plt.legend()
                    plt.savefig(result_dir + 'Variance2_%d.png'%iter)
                    plt.show()
                    plt.close()
                # sava data2
                if True:
                    np.savez(result_dir + '2',
                             X_l=model2.X_l,
                             X_h=model2.X_h,
                             y_train_l=model2.Y_l,
                             y_train_h=model2.Y_h,

                             # hyp_l=model.modelLf.kriging.hyperParams,
                             AF_h=mes_h2,
                             AF_h_l=mes_h_l,
                             RMSE=result2['RMSE'],
                             R_square=result2['R_square'],
                             IR=result2['IR'],
                             SR=result2['SR'],
                             total_cost=result2['total_cost'],
                             # AF_h_l=mes_h_l,
                             )

                    df_list = []
                    sheet_list = []
                    try:
                        df_result = pd.DataFrame({'RMSE': result2['RMSE'],
                                                  'R_square': result2['R_square'],
                                                  'IR': result2['IR'],
                                                  'SR': result2['SR'],
                                                  'total_cost': result2['total_cost'],
                                                  })
                        df_list.append(df_result)
                        sheet_list.append('_result')
                    except:
                        print('Failed to save result into excel.')
                    try:
                        df_train_h = pd.DataFrame({'xTrain_l': model2.X_l.ravel(),
                                                   'yTrain_l': model2.Y_l.ravel(), })
                        df_list.append(df_train_h)
                        sheet_list.append('LF Train')
                    except:
                        print('Failed to save LF Train into excel.')
                    try:
                        df_train_h = pd.DataFrame({'xTrain_h': model2.X_h.ravel(),
                                                   'yTrain_h': model2.Y_h.ravel(), })
                        df_list.append(df_train_h)
                        sheet_list.append('HF Train')
                    except:
                        print('Failed to save HF Train into excel.')


                    try:
                        df_af = pd.DataFrame({'AF_h': mes_h2.ravel(),
                                              'AF_h_l': mes_h_l.ravel(),
                                              # 'AF_h_l': mes_h_l.ravel(),
                                              })
                        df_list.append(df_af)
                        sheet_list.append('_AF')
                    except:
                        print('Failed to save AF into excel.')


                    with pd.ExcelWriter(result_dir + '2.xlsx') as writer:
                        for i in range(len(df_list)):
                            df_list[i].to_excel(writer, sheet_name=sheet_list[i], index=1)
                # pre opt model2
                if True:
                    y_best_set2 = Opt_Random_Sample(model=model2,
                                                    input_dim=input_dim,
                                                    Size=100,
                                                    RNG=np.random.default_rng(seed),
                                                    maximize=maximize)

                    # f(HF)
                    mes_h2 = mes_compute(y_best_set2,
                                         mean=mean_h2,
                                         std=std_h2,
                                         maximize=maximize)
                    # res_h = mes_h / cost  # /3

                    # f(HF) - f(HF-LF)
                    var2 = std_h2 ** 2 - rho2 ** 2 * std_l2 ** 2
                    std2 = np.zeros((len(var2), 1))
                    for j in range(len(var2)):
                        std2[j] = np.sqrt(var2[j]) if var2[j] > 0 else std_h2[j]
                    mes_h_on_l2 = mes_compute(y_best_set2,
                                              mean=mean_h2,
                                              std=std2,
                                              # cov_l_h,
                                              maximize=maximize, )
                    mes_h_l = mes_h2 - mes_h_on_l2
                if np.max(mes_h2)<=0 and np.max(mes_h_l)<=0:
                    break
                # iter += 1
        one_exp(bench_name, seed=0)

from plotResult_func import plotResult1
if __name__ == '__main__':

    for cost_h in [5.0,2.0,10.0]:
        path = 'C:\\Users\\Teo\\Desktop\\2\\test_22\\'
        main()

        measure_list = ['SR',
                        'IR',
                        'RMSE',
                        'R_square']
        plotResult1(path=path+'cost%d_'%cost_h,
                    measure_list=measure_list)