from CoKriging_func import CoKriging as MF
from benchmark_func import *
import matplotlib.pyplot as plt
from pyDOE import lhs
import numpy as np
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
from matplotlib import cm
import pandas as pd
'''VFEI抽样过程'''
# 1.设置高低精度函数，模型初始参数
result_dir = 'C:\\Users\\Teo\\Desktop\\补充\\sample_VFEI00\\'
f_h = FO
f_l = FOa_low
input_dim = 1
grid_size = 100
maximize = 0
n_init_l = 5
n_init_h = 3
iter = 10

# x_l = lhs(n=input_dim,
#                samples=n_init_l)
# x_h = lhs(n=input_dim,
#                samples=n_init_h)
# x_l = np.array([[0.1],[0.25],[0.44],[0.65],[0.97]])
# x_h = np.array([[0.17], [0.5],[0.78]])
x_l = np.array([[0], [0.17], [0.37],[0.42],[0.76],[0.96]])
x_h = np.array([ [0.22], [0.62],[0.98]])
y_l = f_l(x_l)
y_h = f_h(x_h)

from sklearn.utils.extmath import cartesian
grid_1d = np.linspace(0, 1, grid_size)
x_pool = cartesian([grid_1d for _ in range(input_dim)])
y_pool = f_h(x_pool)

# 2.建立模型
model = MF(x_l, y_l, x_h, y_h, dim=input_dim)
model.fitModel()

for k in range(iter):
    # 3.画图，展现抽样过程
    #
    # 3.1 计算模型效果
    mean_h = model.meanPredict(x_pool)
    mean_l = model.meanLowPredict(x_pool)
    std_h = model.stdPredict(x_pool)
    std_l = model.stdLowPredict(x_pool)
    rho = model.rho

    # 3.2 计算习得函数，并优化
    from BO_compute_func import ei_compute
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



    # 3.3 画图
    # 3.3.1 画布设置
    fig = plt.figure(dpi=300, figsize=(16,8))
    a1 = HostAxes(fig, [0.05, 0.05, 0.85, 0.9])
    a2 = ParasiteAxes(a1, sharex=a1)
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    colorset = [cm.Reds(0.5), cm.Blues(0.8), cm.BuGn(0.5),  cm.Purples(0.5)]

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

    # 3.3.2 画模型响应图
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

    # 3.3.2 画习得函数图
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
               s=1000,
               c='b')  # , label='LF sampled points'

    a2.scatter(next_x, next_x_af_value,
               label='next point', marker='o'if next_fidelity=='HF' else '*',s=800,
               c='r')


    # a1.legend()

    plt.savefig(result_dir + 'VFEI%d.png'%k)
    plt.show()
    plt.close()

    # 4.更新模型
    model.addSample(next_x, next_y, next_fidelity)

np.savez(result_dir+'result',
        X_l=model.X_l,
        X_h=model.X_h,
        rou=model.rho)
df_LF = pd.DataFrame({'X_l': model.X_l.ravel(),
                      'Y_l': model.Y_l.ravel()})

df_HF = pd.DataFrame({'X_h': model.X_h.ravel(),
                      'Y_h': model.Y_h.ravel()})

df_list = [df_LF, df_HF]
sheet_list = ['LF', 'HF']
with pd.ExcelWriter(result_dir + 'result_%s.xlsx'% str(model.rho)) as writer:
    for i in range(len(df_list)):
        df_list[i].to_excel(writer, sheet_name=sheet_list[i], index=1)