from CoKriging_new_func import CoKriging as MF
from benchmark_func import *
import matplotlib.pyplot as plt
from pyDOE import lhs
import numpy as np
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
from matplotlib import cm
import pandas as pd
from BO_compute_func import mes_compute, Opt_Random_Sample, dmes_compute
from BO_compute_func import ei_compute
from Test_func import RMSE
from joblib import delayed, Parallel
import itertools as it

'''VFEI抽样过程'''
# 1.设置高低精度函数，模型初始参数
path = 'C:\\Users\\Teo\\Desktop\\补充\\sample_VS_new\\'
f_h = FO
f_l = FOa_low
input_dim = 1
grid_size = 100
maximize = 0
n_init_l = 5
n_init_h = 3
iter = 15

from sklearn.utils.extmath import cartesian
grid_1d = np.linspace(0, 1, grid_size)
x_pool = cartesian([grid_1d for _ in range(input_dim)])
y_pool = f_h(x_pool)

# x_l = lhs(n=input_dim,
#                samples=n_init_l)
# x_h = lhs(n=input_dim,
#                samples=n_init_h)
# x_l = np.array([[0.1],[0.25],[0.44],[0.65],[0.97]])
# x_h = np.array([[0.17], [0.5],[0.78]])





def model_test(model, mean, y,
               rmse_list, cost_list,
               ir_list, sr_list, cost):
    y_opt = np.max(y) if maximize else np.min(y)
    y_best = np.max(model.Y_h) if maximize else np.min(model.Y_h)

    best_i = np.argmax(mean) if maximize else np.argmin(mean)
    f_x_best = y[best_i]

    ir = float(y_opt - f_x_best)
    sr = float(y_opt - y_best)
    rmse = RMSE(y, mean)
    total_cost = float(len(model.X_h) + len(model.X_l)/cost)

    rmse_list.append(rmse)
    cost_list.append(total_cost)
    ir_list.append(ir)
    sr_list.append(sr)
    return

try:
    data = np.load('C:\\Users\\Teo\\Desktop\\试验结果\\test_21\\cost5_0_00.npz')
    x_l = data['X_l'][:n_init_l]
except:
    x_l = lhs(n=input_dim,
                   samples=n_init_l)

idx = np.array([[0,1,2],
                [0,1,3],
                [0,1,4],
                [1,2,3],
                [1,2,4],
                [2,3,4]])
# for j in range(len(idx)):
#     x_h = x_l[idx[j,:]]
# for cost in [2, 5, 10]:

def exp(cost, j):
    x_h = x_l[idx[j, :]]
    y_l = f_l(x_l)
    y_h = f_h(x_h)
    result_dir = path + 'CR%s_%d_' % (str(cost), j)
    if True:
        # 2. VFEI优化
        # 2.1 建立模型
        model = MF(x_l, y_l, x_h, y_h, dim=input_dim)
        model.fitModel()

        rmse_list, cost_list, ir_list, sr_list = [], [], [],[]

        for k in range(iter):

            # 2.2 计算模型效果
            mean_h = model.meanPredict(x_pool)
            mean_l = model.meanLowPredict(x_pool)
            std_h = model.stdPredict(x_pool)
            std_l = model.stdLowPredict(x_pool)
            model_test(model, mean_h, y_pool, rmse_list, cost_list, ir_list, sr_list, cost)

            # 2.3 计算习得函数，并优化
            y_best = np.max(model.Y_h) if maximize else np.min(model.Y_h)
            af_h = ei_compute(mean=mean_h,
                              std=std_h,
                              y_best=y_best,
                              x0=0,
                              maximize=maximize)
            af_l = ei_compute(mean=mean_h,
                              std=abs(std_l * model.rho),
                              y_best=y_best,
                              x0=0,
                              maximize=maximize)
            # y_best_set = Opt_Random_Sample(model=model,
            #                                input_dim=input_dim,
            #                                Size=100,
            #                                RNG=np.random.default_rng(0),
            #                                maximize=maximize)
            # af_h = mes_compute(mean=mean_h,
            #                    std=std_h,
            #                    y_best_set=y_best_set,
            #                    maximize=maximize)
            # af_l = mes_compute(mean=mean_h,
            #                    std=abs(std_l * model.rho),
            #                    y_best_set=y_best_set,
            #                    maximize=maximize)

            i = -1
            while abs(i) < len(af_h) and any(np.allclose(x_pool[np.argsort(af_h)[i]], model.X_h[j]) for j in
                                             range(len(model.X_h))):
                i = i - 1
            max_i_h = np.argsort(af_h)[i]

            i = -1
            while abs(i) < len(af_l) and any((np.allclose(x_pool[np.argsort(af_l)[i]], model.X_l[j]) for j in
                                              range(len(model.X_l)))):
                i = i - 1
            max_i_l = np.argsort(af_l)[i]

            if af_h[max_i_h] >= af_l[max_i_l]:
                next_fidelity = 'HF'
                max_i = max_i_h
                # next_i = max_i
                next_x = x_pool[max_i]
                next_x_af_value = af_h[max_i]
                next_y = f_h(next_x)

            elif af_h[max_i_h] < af_l[max_i_l]:
                # if af_l[max_i_l]>0:
                next_fidelity = 'LF'
                max_i = max_i_l

                next_x = x_pool[max_i]
                next_x_af_value = af_l[max_i]
                next_y = f_l(next_x)

            # 2.4 画图
            # 2.4.1 画布设置
            fig = plt.figure(dpi=300, figsize=(16, 8))
            a1 = HostAxes(fig, [0.05, 0.05, 0.85, 0.9])
            a2 = ParasiteAxes(a1, sharex=a1)
            plt.rcParams['font.family'] = 'Times New Roman'
            plt.rcParams['font.size'] = 12
            colorset = [cm.Reds(0.5), cm.Blues(0.8),cm.BuGn(0.5),  cm.Purples(0.5)]

            a1.parasites.append(a2)  # append axes
            a1.axis['right'].set_visible(False)  # invisible right axis of ax_cof
            a1.axis['top'].set_visible(False)
            a2.axis['right'].set_visible(True)
            a2.axis['right'].major_ticklabels.set_visible(True)
            a2.axis['right'].label.set_visible(True)

            fig.add_axes(a1)
            a1.set_ylim(-25, 20)  # set limit of x, y
            a2.set_ylim(0, 5 * abs(next_x_af_value))

            # set label for axis
            a1.set_ylabel('Y')
            a1.set_xlabel('X')
            a2.set_ylabel(r'$\alpha(x)$')

            # 2.4.2 画模型响应图
            a1.plot(x_pool,
                    y_pool,
                    label=r'y(x)', linewidth=3, color='r')

            a1.plot(x_pool,
                    mean_h,
                    # linestyle='-.',
                    label=r'$f_2(x)$',
                    linewidth=5, color='black'
                    )
            a1.plot(x_pool,
                    mean_l,
                    # linestyle=':',
                    label=r'$f_1(x)$',
                    linewidth=5, color='gray'
                    )

            a1.fill_between(np.squeeze(x_pool),
                            np.squeeze(mean_h + 3 * std_h),
                            np.squeeze(mean_h - 3 * std_h),
                            alpha=0.3,
                            label=r'confidence($f_2$,99.7% )',
                            color=colorset[1])
            plt.fill_between(np.squeeze(x_pool),
                             np.squeeze(mean_l + 3 * std_l),
                             np.squeeze(mean_l - 3 * std_l),
                             alpha=0.3,
                             label=r'confidence($f_1$,99.7% )',
                             color=colorset[2])

            # 2.4.3 画习得函数图
            # a2 = fig.add_axes([0,0,0.9,0.9])
            a2.plot(x_pool, af_h, label=r'$\alpha_2(x)$',
                    linestyle='-.',
                    color=colorset[1], linewidth=5)
            a2.plot(x_pool, af_l, label=r'$\alpha_1(x)$',
                    linestyle='-.',
                    color=colorset[2], linewidth=5)

            a1.scatter(model.X_h,
                       model.Y_h,
                       label=r'$y_h$  : %d' % len(model.X_h),
                       marker='o',
                       s=800,
                       c='b')  # , label='HF sampled points'
            a1.scatter(model.X_l,
                       model.Y_l,
                       label=r'$y_l$  : %d' % len(model.X_l),
                       marker='*',
                       s=800,
                       c='b')  # , label='LF sampled points'

            a2.scatter(next_x, next_x_af_value,
                       label='next point', marker='o' if next_fidelity == 'HF' else '*', s=800,
                       c='r')

            # a1.legend()

            plt.savefig(result_dir + 'VFMES%d.png' % k)
            plt.show()
            plt.close()

            # 2.5 更新模型
            model.addSample(next_x, next_y, next_fidelity)
        # 2.6 存储VFMES数据
        np.savez(result_dir + 'VFMESresult',
                 X_l=model.X_l,
                 X_h=model.X_h,
                 rou=model.rho,
                 SR=sr_list,
                 IR=ir_list,
                 RMSE=rmse_list,
                 cost=cost_list)

        df_LF = pd.DataFrame({'X_l': model.X_l.ravel(),
                              'Y_l': model.Y_l.ravel()})

        df_HF = pd.DataFrame({'X_h': model.X_h.ravel(),
                              'Y_h': model.Y_h.ravel()})
        df_test = pd.DataFrame({'SR': sr_list,
                              'IR':ir_list,
                              'RMSE':rmse_list,
                              'cost':cost_list})

        df_list = [df_LF, df_HF, df_test]
        sheet_list = ['LF', 'HF', 'test']
        with pd.ExcelWriter(result_dir + 'VFMESresult_%s.xlsx' % str(model.rho)) as writer:
            for i in range(len(df_list)):
                df_list[i].to_excel(writer, sheet_name=sheet_list[i], index=1)
    # 3. DMES优化
    if True:
        # 3.1 建立模型
        model = MF(x_l, y_l, x_h, y_h, dim=input_dim)
        model.fitModel()
        rmse_list, cost_list, ir_list, sr_list = [], [], [],[]
        for k in range(iter):

            # 3.2 计算模型效果
            mean_h = model.meanPredict(x_pool)
            mean_l = model.meanLowPredict(x_pool)
            std_h = model.stdPredict(x_pool)
            std_l = model.stdLowPredict(x_pool)
            rho = model.rho
            model_test(model, mean_h, y_pool,
                 rmse_list, cost_list,
                 ir_list, sr_list, cost)

            # 3.3 计算习得函数，并优化
            y_best_set = Opt_Random_Sample(model=model,
                                            input_dim=input_dim,
                                            Size=100,
                                            RNG=np.random.default_rng(0),
                                            maximize=maximize)
            af_h, af_l = dmes_compute(y_best_set=y_best_set,
                         mean_h=mean_h,
                         std_h=std_h,
                         std_l=std_l,
                         rho=model.rho,
                         cost=cost,
                         maximize=maximize)


            i = -1
            while abs(i) < len(af_h) and any(np.allclose(x_pool[np.argsort(af_h)[i]], model.X_h[j]) for j in
                                             range(len(model.X_h))):
                i = i - 1
            max_i_h = np.argsort(af_h)[i]

            i = -1
            while abs(i) < len(af_l) and any((np.allclose(x_pool[np.argsort(af_l)[i]], model.X_l[j]) for j in
                                              range(len(model.X_l)))):
                i = i - 1
            max_i_l = np.argsort(af_l)[i]

            if af_h[max_i_h] >= af_l[max_i_l]:
                next_fidelity = 'HF'
                max_i = max_i_h
                # next_i = max_i
                next_x = x_pool[max_i]
                next_x_af_value = af_h[max_i]
                next_y = f_h(next_x)

            elif af_h[max_i_h] < af_l[max_i_l]:
                # if af_l[max_i_l]>0:
                next_fidelity = 'LF'
                max_i = max_i_l

                next_x = x_pool[max_i]
                next_x_af_value = af_l[max_i]
                next_y = f_l(next_x)



            # 3.4 画图
            # 3.4.1 画布设置
            fig = plt.figure(dpi=300, figsize=(16,8))
            a1 = HostAxes(fig, [0.05, 0.05, 0.85, 0.9])
            a2 = ParasiteAxes(a1, sharex=a1)
            plt.rcParams['font.family'] = 'Times New Roman'
            plt.rcParams['font.size'] = 12
            colorset = [cm.Reds(0.5), cm.BuGn(0.5), cm.Blues(0.8), cm.Purples(0.5)]

            a1.parasites.append(a2) # append axes
            a1.axis['right'].set_visible(False) #invisible right axis of ax_cof
            a1.axis['top'].set_visible(False)
            a2.axis['right'].set_visible(True)
            a2.axis['right'].major_ticklabels.set_visible(True)
            a2.axis['right'].label.set_visible(True)

            fig.add_axes(a1)
            a1.set_ylim(-25 , 20) #set limit of x, y
            a2.set_ylim(0, 5*abs(next_x_af_value))

            #set label for axis
            a1.set_ylabel('Y')
            a1.set_xlabel('X')
            a2.set_ylabel(r'$\alpha(x)$')

            # 3.4.2 画模型响应图
            a1.plot(x_pool,
                     y_pool,
                     label=r'y(x)',linewidth=3, color='r')

            a1.plot(x_pool,
                     mean_h,
                     # linestyle='-.',
                     label=r'$f_2(x)$',
                     linewidth=5,color='black'
                     )
            a1.plot(x_pool,
                     mean_l,
                     # linestyle=':',
                     label=r'$f_1(x)$',
                    linewidth=5, color='gray'
                     )

            a1.fill_between(np.squeeze(x_pool),
                             np.squeeze(mean_h + 3 * std_h),
                             np.squeeze(mean_h - 3 * std_h),
                             alpha=0.3,
                             label=r'confidence($f_2$,99.7% )',
                            color=colorset[1])
            plt.fill_between(np.squeeze(x_pool),
                             np.squeeze(mean_l + 3 * std_l),
                             np.squeeze(mean_l - 3 * std_l),
                             alpha=0.3,
                             label=r'confidence($f_1$,99.7% )',
                             color=colorset[2])

            # 3.4.3 画习得函数图
            # a2 = fig.add_axes([0,0,0.9,0.9])
            a2.plot(x_pool, af_h, label=r'$\alpha_2(x)$',
                    linestyle='-.',
                    color=colorset[1], linewidth=5)
            a2.plot(x_pool, af_l, label=r'$\alpha_1(x)$',
                    linestyle='-.',
                    color=colorset[2], linewidth=5)

            a1.scatter(model.X_h,
                        model.Y_h,
                        label=r'$y_h$  : %d' % len(model.X_h),
                        marker='o',
                       s=800,
                       c='b')  # , label='HF sampled points'
            a1.scatter(model.X_l,
                        model.Y_l,
                        label=r'$y_l$  : %d' % len(model.X_l),
                        marker='*',
                       s=800,
                       c='b')  # , label='LF sampled points'

            a2.scatter(next_x, next_x_af_value,
                       label='next point', marker='o'if next_fidelity=='HF' else '*',s=800,
                       c='r')


            # a1.legend()

            plt.savefig(result_dir + 'DMES%d.png'%k)
            plt.show()
            plt.close()

            # 3.5 更新模型
            model.addSample(next_x, next_y, next_fidelity)

        # 3.6 存储DMES数据
        np.savez(result_dir+'DMESresult',
                X_l=model.X_l,
                X_h=model.X_h,
                rou=model.rho)
        df_LF = pd.DataFrame({'X_l': model.X_l.ravel(),
                              'Y_l': model.Y_l.ravel()})

        df_HF = pd.DataFrame({'X_h': model.X_h.ravel(),
                              'Y_h': model.Y_h.ravel()})

        df_test = pd.DataFrame({'SR': sr_list,
                                'IR': ir_list,
                                'RMSE': rmse_list,
                                'cost': cost_list})

        df_list = [df_LF, df_HF, df_test]
        sheet_list = ['LF', 'HF', 'test']

        with pd.ExcelWriter(result_dir + 'DMESresult_%s.xlsx'% str(model.rho)) as writer:
            for i in range(len(df_list)):
                df_list[i].to_excel(writer, sheet_name=sheet_list[i], index=1)

Parallel(n_jobs=-3, verbose=10)([delayed(exp)(cost, j)
                                         for cost, j in
                                         it.product([2, 5, 10], [0, 1, 2, 3, 4, 5])])
