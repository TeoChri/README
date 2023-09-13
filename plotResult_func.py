import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plotResult(path, bench_name_list, af_name_list, seed_list, measure_list):
    line_list = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    mark_list = ['X', '^', '*', 'd', 'o', 'X', '^', '*', 'd', 'o']
    for bench_name in bench_name_list:
        # 计算不同的seed下的结果。
        for seed in seed_list:
            # 存储不同优化准则下的结果。
            for measure in measure_list:
                exsist = False
                # 对不同的习得函数的采样效果分别计算结果。
                for j in range(len(af_name_list)):
                    try:
                        cost = np.load(path + '%s_%s_seed%d.npz' % (bench_name, af_name_list[j], seed))['total_cost']
                        result = np.load(path + '%s_%s_seed%d.npz' % (bench_name, af_name_list[j], seed))[measure]
                        exsist = True

                        plt.plot(cost,
                                 result,
                                 label=af_name_list[j],
                                 linestyle=line_list[j],
                                 marker=mark_list[j],
                                 markersize=10-j)

                        initial_result = result[0]
                        delta = 0.01

                        if measure == 'R_square':
                            lim = [initial_result, 1]
                        else:
                            lim = [0, initial_result] if initial_result > 0 else [initial_result, 0]
                        lim[0] = lim[0] - delta
                        lim[1] = lim[1] + delta
                    except:
                        pass
                    # plt.plot(cost, np.log(abs(result)), label=af_name_list[j])

                if exsist:
                    plt.xlabel('Total cost')
                    plt.ylabel(measure)
                    # plt.xlim([0, 500])
                    # plt.ylim([-1,1])
                    try:
                        # plt.ylim(lim)
                        pass
                    except:
                        pass
                    plt.title('%s' % (bench_name))
                    plt.legend()
                    plt.savefig(path + 'Result_%s_seed%d_%s.png' % (bench_name, seed, measure))
                    plt.show()
                    plt.close()
    print('Result is plotted.')


def plotResult1(path,  measure_list):
    line_list = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    mark_list = ['X', '^', '*', 'd', 'o', 'X', '^', '*', 'd', 'o']
    for i in range(6):
        # 计算不同的采样下的结果。
        for measure in measure_list:
        # 对不同的习得函数的采样效果分别计算结果。
            for j in range(1,3):
                # cost = np.load(path + '%d_%d.npz' % (i,j))['total_cost']
                # result = np.load( path + '%d_%d.npz' % (i,j))[measure]

                data = pd.read_excel(path + '%d_%d.xlsx' % (i,3-j))
                cost = data['total_cost']
                result = data[measure]

                plt.plot(cost,
                         result,
                         label='DMES' if j==1 else 'VFMES',
                         linestyle=line_list[j],
                         marker=mark_list[j],
                         markersize=10-j)
                # plt.plot(cost, np.log(abs(result)), label=af_name_list[j])

            plt.xlabel('Total cost')
            plt.ylabel(measure)
            # plt.xlim([0, 500])
            # plt.ylim([-1,1])
            # plt.title('%s, seed=%d' % (bench_name, seed))
            plt.legend()
            plt.savefig(path + 'Result_%d_%s.png' % (i, measure))
            plt.show()
            plt.close()
    print('Result is plotted.')