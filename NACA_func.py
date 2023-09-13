import numpy as np
import pandas as pd
import matplotlib

# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def plotAll():
    path = 'C:\\Users\\Teo\\Desktop\\0504\\NACA0012\\'

    fig1, ax1 = plt.subplots(figsize=(7, 7))
    fig2, ax2 = plt.subplots(figsize=(7, 7))
    fig3, ax3 = plt.subplots(figsize=(7, 7))
    fig4, ax4 = plt.subplots(figsize=(7, 7))

    for i in range(120):
        idx_name = str(i).zfill(3)
        filename = path + 'n%sm' % (idx_name)
        hist_data = pd.read_csv(filename + '.csv', skipinitialspace=True)
        hist_data.columns = hist_data.columns.str.strip()

        x0 = hist_data['0'].values
        x1 = hist_data['1'].values
        y0 = hist_data['2'].values
        y1 = hist_data['3'].values

        if i == 0:
            y0_list = y0.reshape(-1, 1)
            y1_list = y1.reshape(-1, 1)
        else:
            y0_list = np.hstack((y0_list, y0.reshape(-1, 1)))
            y1_list = np.hstack((y1_list, y1.reshape(-1, 1)))

        ax1.plot(x0, y0)
        ax2.plot(x0, y1)
        ax3.plot(x1, y0)
        ax4.plot(x1, y1)

    # std0 = np.std(y0_list, axis=1)
    # imax0 = np.argmax(std0)
    # imax00 = np.argsort(std0)[-2:]
    #
    # std1 = np.std(y1_list, axis=1)
    # imax1 = np.argmax(std1)
    # imax11 = np.argsort(std1)[-2:]

    # ax1.scatter(x0[imax0], y0[imax0], linewidths=18)
    # ax1.scatter(x0[158], y0[158], linewidths=28)
    # ax1.scatter(x0[88], y0[88], linewidths=28)
    # ax1.scatter(x0[154], y0[154], linewidths=28)
    ax1.set_xlabel('X0')
    ax1.set_ylabel(r'$C_P$')
    ax1.legend()
    fig1.show()
    fig1.savefig(path + 'fig1.png')

    # ax2.scatter(x0[imax1], y1[imax1], linewidths=18)
    # ax2.scatter(x0[imax11], y1[imax11], linewidths=8)
    # ax2.scatter(x0[158], y1[158], linewidths=28)
    # ax2.scatter(x0[88], y1[88], linewidths=28)
    # ax2.scatter(x0[154], y1[154], linewidths=28)
    ax2.set_xlabel('X0')
    ax2.set_ylabel(r'$C_P(X)$')
    ax2.legend()
    fig2.show()
    fig2.savefig
    fig2.savefig(path + 'fig2.png')

    # ax3.scatter(x1[imax0], y0[imax0], linewidths=18)
    # ax3.scatter(x0[imax00], y0[imax00], linewidths=18)
    # ax3.scatter(x1[158], y0[158], linewidths=38)
    # ax3.scatter(x1[88], y0[88], linewidths=38)
    # ax3.scatter(x1[154], y0[154], linewidths=38)
    ax3.set_xlabel('X1')
    ax3.set_ylabel(r'$C_P$')
    ax3.legend()
    fig3.show()
    fig3.savefig(path + 'fig3.png')

    # ax4.scatter(x1[imax1], y1[imax1], linewidths=18)
    # ax4.scatter(x1[158], y1[158], linewidths=38)
    # ax4.scatter(x1[88], y1[88], linewidths=38)
    # ax4.scatter(x1[154], y1[154], linewidths=38)
    ax4.set_xlabel('X1')
    ax4.set_ylabel(r'$C_P(X)$')
    ax4.legend()
    fig4.show()
    fig4.savefig(path + 'fig4.png')
    return


def modelCompare():
    from Kriging_func import Kriging
    from CoKriging_func import CoKriging
    from expNACA import expNACA, expAllNACA
    from Test_func import RMSE

    i_loc = 158
    x_pool = pd.read_excel('C:\\Users\\Teo\\Desktop\\0504\\NACA0012\\m6_120samples_9input.xlsx').values
    y_pool = expAllNACA(fidelity='HF', i_loc=i_loc)
    y_pool_l = expAllNACA(fidelity='LF', i_loc=i_loc)

    # model = Kriging(x_pool, y_pool, dim=9, Rtype=11)
    # model.fitModel()
    #
    # mean = model.meanPredict(x_pool)
    # std = model.stdPredict(x_pool)
    #
    # rmse = RMSE(y_pool, mean)
    # plt.plot(np.arange(len(x_pool)), y_pool, label = 'True')
    # plt.plot(np.arange(len(x_pool)), mean, label='model')
    # plt.fill_between(np.arange(len(x_pool)), np.squeeze(mean + 3*std), np.squeeze(mean - 3*std), alpha=0.3 )
    # plt.legend()
    # plt.show()

    idx_h = pd.read_excel('C:\\Users\\Teo\\Desktop\\0504\\NACA02\\10_1_NACA_DMES_seed0.xlsx', sheet_name='Idx_h')[
        'Idx_h'].values
    idx_l = pd.read_excel('C:\\Users\\Teo\\Desktop\\0504\\NACA02\\10_1_NACA_DMES_seed0.xlsx', sheet_name='Idx_l')[
        'Idx_l'].values

    X_h = x_pool[idx_h]
    Y_h = y_pool[idx_h]

    X_l = x_pool[idx_l]
    Y_l = y_pool_l[idx_l]

    model_l = Kriging(X_h, Y_h, dim=9, Rtype=11)
    model_l.fitModel()
    mean_l = model_l.meanPredict(x_pool)
    rmse_l = RMSE(y_pool, mean_l)

    model_h = CoKriging(X_l, Y_l, X_h, Y_h, dim=9, Rtype_h=11, Rtype_l=11)
    model_h.fitModel()
    mean_h = model_h.meanPredict(x_pool)
    rmse_h = RMSE(y_pool, mean_h)

    model_l0 = Kriging(X_h, Y_h, dim=9, Rtype=6)
    model_l0.fitModel()
    mean_l0 = model_l0.meanPredict(x_pool)
    rmse_l0 = RMSE(y_pool, mean_l0)

    model_h0 = CoKriging(X_l, Y_l, X_h, Y_h, dim=9, Rtype_h=6, Rtype_l=6)
    model_h0.fitModel()
    mean_h0 = model_h0.meanPredict(x_pool)
    rmse_h0 = RMSE(y_pool, mean_h0)
    return


def addKMES():
    from expNACA import expAllNACA
    from SFBO_func import SFBO
    from plotResult_func import plotResult
    cost = 5
    iter = 0
    path = 'C:\\Users\\Teo\\Desktop\\0504\\NACA03\\'

    # i_loc = 157
    # x_pool = pd.read_excel('C:\\Users\\Teo\\Desktop\\0504\\NACA0012\\m6_120samples_9input.xlsx').values
    # y_pool = expAllNACA(fidelity='HF', i_loc=i_loc)
    #
    # idx_h = pd.read_excel('C:\\Users\\Teo\\Desktop\\0504\\NACA03\\5_0_NACA_DMES_seed0.xlsx', sheet_name='Idx_h')[
    #     'Idx_h'].values[:3].reshape(-1, 1)
    # X = x_pool[idx_h]
    # Y = y_pool[idx_h]
    #
    # bo = SFBO(result_dir=path + '%d_%d_' % (cost, iter),
    #           input_dim=9,
    #           benchmark_name='NACA',
    #           x_pool=x_pool,
    #           y_pool=y_pool,
    #           idx_h=idx_h,
    #           x_train_h=X,
    #           y_train_h=Y,
    #           af_name='K-MES',
    #           seed=0,
    #           cost=cost,
    #           maximize=1)
    #
    # while bo.total_cost[-1] < 50:
    #     bo.nextObservation()
    # bo.saveResult()

    measure_list = [
        'SR',
        'IR',
        'RMSE',
        'R_square',
    ]
    af_name_list = [
        'DMES',
        'VFEI',
        'EFI',
        'MES',
        'K-MES'
    ]
    plotResult(path=path + '%d_%d_' % (cost, iter),
               bench_name_list=['NACA'],
               af_name_list=af_name_list,
               seed_list=[0],
               measure_list=measure_list)


from Kriging_func import Kriging
from CoKriging_func import CoKriging
from expRAE_func import expResult, expAllResult

grid_size = 20
x_mesh = pd.read_csv('C:\\Users\\Teo\\Desktop\\1\\grid_size%d\\exp_HF.csv' % grid_size, skipinitialspace=True)
x_mesh.columns = x_mesh.columns.str.strip()
x_pool = x_mesh[["AOA", "MACH_NUMBER"]].values
y_pool = expAllResult(fidelity='HF', grid_size=grid_size)

X_l = np.array([4.960526316, 0.747368421,
                4.328947368, 0.707894737,
                4.486842105, 0.739473684,
                4.802631579, 0.744736842,
                4.644736842, 0.736842105,
                5.671052632, 0.715789474,
                5.355263158, 0.723684211,
                5.118421053, 0.728947368,
                5.355263158, 0.7,
                5.197368421, 0.710526316,
                5.75, 0.742105263,
                4.644736842, 0.731578947,
                5.513157895, 0.718421053,
                4.25, 0.721052632,
                4.960526316, 0.702631579,
                ]).reshape(-1, 2)
Y_l = np.array([0.977263822,
                1.088235,
                0.999154418,
                0.978269997,
                1.018151183,
                1.124263199,
                1.091277677,
                1.06641211,
                1.188370542,
                1.161169061,
                1.036755676,
                1.053908798,
                1.115576295,
                1.080316652,
                1.155665421,
                ]).reshape(-1, 1)
X_h = np.array([5.118421053, 0.718421053,
                4.723684211, 0.728947368,
                5.592105263, 0.731578947,
                4.407894737, 0.707894737,
                5.197368421, 0.742105263,
                ]).reshape(-1, 2)
Y_h = np.array([1.065339075,
                0.997955635,
                0.96792212,
                1.024719544,
                0.948869089,
                ]).reshape(-1, 1)
model0 = CoKriging(X_l, Y_l, X_h, Y_h, 2, 11)
model0.fitModel()
mean0 = model0.meanPredict(x_pool)

model1 = Kriging(X_h, Y_h, 2, 11)
model1.fitModel()
mean1 = model1.meanPredict(x_pool)

size = grid_size
X = x_pool[:, 0].reshape(size, size).T
Y = x_pool[:, 1].reshape(size, size).T
z_h_mesh = y_pool.reshape(size, size).T

z0_mesh = mean0.reshape(size, size).T
z1_mesh = mean1.reshape(size, size).T

fig1 = plt.figure(figsize=[6,6])
ax1 = fig1.add_subplot(1, 1, 1, projection='3d')
# ax1.contour3D(X, Y, z_h_mesh, 50, label='HF')
ax1.plot_surface(X, Y, z_h_mesh,
                 alpha=0.8,
                 cmap=plt.get_cmap('viridis'), # rainbow, viridis
                 antialiased=True)
# ax1.plot_surface(X, Y, z_l_mesh, alpha=0.4, color='aqua')
# ax1.scatter3D(X_h[:, 0],
#              X_h[:, 1],
#              Y_h,
#              label='HF samples',
#              marker='o',
#              color='r',
#               linewidths=10,
#              )
# ax1.scatter3D(X_l[:, 0],
#              X_l[:, 1],
#              Y_l,
#              label='LF samples',
#              marker='*',
#              color='orange'
#              )

ax1.view_init(elev=28, azim=160)
ax1.set_xlabel('AOA')
ax1.set_ylabel('MACH')
ax1.set_zlabel(r'$C_L$')
ax1.set_zlim(0.9, 1.2)
ax1.legend()
fig1.show()
fig1.savefig('C:\\Users\\Teo\\Desktop\\试验结果\\RAE2822_simulation')
plt.close()

fig2 = plt.figure(figsize=[6,6])
ax2 = fig2.add_subplot(1, 1, 1, projection='3d')
# ax1.contour3D(X, Y, z_h_mesh, 50, label='HF')
ax2.plot_surface(X, Y, z0_mesh,
                 alpha=0.8,
                 cmap=plt.get_cmap('viridis'), # rainbow, viridis
                 antialiased=True)
# ax1.plot_surface(X, Y, z_l_mesh, alpha=0.4, color='aqua')
ax2.scatter3D(X_h[:, 0],
             X_h[:, 1],
             Y_h,
             label='HF samples',
             marker='o',
             color='r',

             )
ax2.scatter3D(X_l[:, 0],
             X_l[:, 1],
             Y_l,
             label='LF samples',
             marker='*',
             color='orange'
             )
ax2.view_init(elev=28, azim=160)
ax2.set_xlabel('AOA')
ax2.set_ylabel('MACH')
ax2.set_zlabel(r'$C_L$')
ax2.set_zlim(0.9, 1.2)
ax2.legend()
fig2.show()
fig2.savefig('C:\\Users\\Teo\\Desktop\\试验结果\\RAE2822_MFmodel')
plt.close()

fig3 = plt.figure(figsize=[6,6])
ax3 = fig3.add_subplot(1, 1, 1, projection='3d')
# ax1.contour3D(X, Y, z_h_mesh, 50, label='HF')
ax3.plot_surface(X, Y, z1_mesh,
                 alpha=0.8,
                 cmap=plt.get_cmap('viridis'), # rainbow, viridis
                 antialiased=True)
# ax1.plot_surface(X, Y, z_l_mesh, alpha=0.4, color='aqua')
ax3.scatter3D(X_h[:, 0],
             X_h[:, 1],
             Y_h,
             label='HF samples',
             marker='o',
             color='r',
             )
ax3.scatter3D(X_l[:, 0],
             X_l[:, 1],
             Y_l,
             label='LF samples',
             marker='*',
             color='orange'
             )

ax3.view_init(elev=28, azim=160)
ax3.set_xlabel('AOA')
ax3.set_ylabel('MACH')
ax3.set_zlabel(r'$C_L$')
ax3.set_zlim(0.9, 1.2)
ax3.legend()
fig3.show()
fig3.savefig('C:\\Users\\Teo\\Desktop\\试验结果\\RAE2822_SFmodel')
plt.close()