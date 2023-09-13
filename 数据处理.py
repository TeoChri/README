import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#
def plotBox():
    '''1.设置测试函数、优化方式名称'''
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
        'RAE2822'
    ]
    af_name_list = ['DMES', 'VFEI', 'EFI', 'MES', ]
    '''2.设置导入、导出数据的路径'''
    out_path = 'C:\\Users\\Teo\\Desktop\\0504\\data\\'
    pathlist = []

    for bench_name in bench_name_list:
        if bench_name == 'RAE2822':
            pathlist.append('C:\\Users\\Teo\\Desktop\\0504\\RAE01\\')
            pathlist.append('C:\\Users\\Teo\\Desktop\\0504\\RAE14\\')
            pathlist.append('C:\\Users\\Teo\\Desktop\\0504\\RAE11\\')
            pathlist.append('C:\\Users\\Teo\\Desktop\\0504\\BO22\\')
        else:
            for i in range(4, 9):
                pathlist.append('C:\\Users\\Teo\\Desktop\\0504\\BO Result0%d\\' % i)

        for cost in [2, 5, 10]:
            # aveCostIR = []
            # aveCostRMSE = []
            # stdCostIR = []
            # stdCostRMSE = []
            for j in range(len(af_name_list)):
                af = af_name_list[j]
                cIR = []
                cCostIR = []
                cRMSE = []
                cCostRMSE = []

                for inpath in pathlist:
                    # 读取收敛时的RMSE，以及RMSE、IR各自收敛时对应的total cost
                    for iter in [0, 1, 2, 3, 4]:
                        try:
                            result_dir = inpath + '%d_%d_%s_%s_seed0.npz' % (cost, iter, bench_name, af)
                            result = np.load(result_dir)
                            IR = result['IR']
                            RMSE = result['RMSE']
                            total_cost = result['total_cost']
                            R_square = result['R_square']

                            # result_dir = inpath + '%d_%d_%s_%s_seed0.xlsx' % (cost, iter, bench_name, af)
                            # result = pd.read_excel(result_dir)
                            # IR = result['%s_seed0_Inference regret'%af].values
                            # RMSE = result['%s_seed0_RMSE'%af].values
                            # total_cost = result['%s_seed0_total cost'%af].values
                            # R_square = result['%s_seed0_Rsquare'%af].values

                            idx = 0
                            while R_square[idx] < 0.9 and abs(RMSE[idx] - np.median(RMSE[:-6])) > 0.01 * abs(
                                    RMSE[0] - np.median(RMSE[:-6])) and idx < len(
                                R_square) - 1:  # abs(RMSE[idx] - np.median(RMSE[:-6])) > 0.01 * (RMSE[0] -np.median(RMSE[:-6]))
                                #  abs(RMSE[idx] - RMSE[-3]) > 0.3 * (max(RMSE) - min(RMSE))
                                idx = idx + 1
                            cRMSE.append(np.min(RMSE[idx:]))
                            cCostRMSE.append(total_cost[idx])

                            idx = 0
                            while abs(IR[idx] - np.median(IR[-3:])) > 0.01 * abs(
                                    IR[0] - np.median(IR[-3:])) and idx < len(
                                    R_square) - 1:  # R_square[idx] < 0.9 and abs(IR[idx] - IR[-3]) > 0.3 * (max(IR) - min(IR))
                                #  abs(IR[idx] - np.median(IR[-3:])) > 0.01 * (IR[0] - np.median(IR[-3:]))
                                idx = idx + 1
                            cIR.append(np.max(IR[idx:]))
                            cCostIR.append(total_cost[idx])
                        except:
                            pass

                # 整合不同af对应的收敛RMSE
                cCostIR = np.asarray(cCostIR).reshape(-1, 1)
                cCostRMSE = np.asarray(cCostRMSE).reshape(-1, 1)
                cRMSE = np.asarray(cRMSE).reshape(-1, 1)
                if j == 0:
                    cAllRMSE = cRMSE
                    cAllCostIR = cCostIR
                    cAllCostRMSE = cCostRMSE
                else:
                    cAllRMSE = np.hstack((cAllRMSE, cRMSE))
                    cAllCostIR = np.hstack((cAllCostIR, cCostIR))
                    cAllCostRMSE = np.hstack((cAllCostRMSE, cCostRMSE))

                # 计算收敛时cost的平均值和方差
                # aveCostIR.append(np.mean(cCostIR))
                # aveCostRMSE.append(np.mean(cCostRMSE))
                # stdCostIR.append(np.std(cCostIR))
                # stdCostRMSE.append(np.std(cCostRMSE))

            # 计算收敛时cost的平均值和方差
            aveCostIR = np.mean(cAllCostIR, axis=0)
            aveCostRMSE = np.mean(cAllCostRMSE, axis=0)
            aveRMSE = np.mean(cAllRMSE, axis=0)
            stdCostIR = np.std(cAllCostIR, axis=0)
            stdCostRMSE = np.std(cAllCostRMSE, axis=0)
            stdRMSE = np.std(cAllRMSE, axis=0)

            # 计算收敛时RMSE比较的得分
            temp = cAllRMSE.argsort(axis=1)
            ranks = temp.argsort(axis=1)
            score = np.zeros(ranks.shape)
            for l in range(ranks.shape[0]):
                for m in range(ranks.shape[1]):
                    if ranks[l, m] == 0:
                        score[l, m] = 1
                    elif ranks[l, m] == 3:
                        score[l, m] = -1
                    else:
                        pass
            total_score = np.sum(score, axis=0)

            ## 保存每次收敛的数据
            np.savez(out_path + '%s_Data_%d.npz' % (bench_name, cost),
                     cAllRMSE=cAllRMSE,
                     cAllCostIR=cAllCostIR,
                     cAllCostRMSE=cAllCostRMSE,
                     aveCostIR=aveCostIR, stdCostIR=stdCostIR,
                     aveCostRMSE=aveCostRMSE, stdCostRMSE=stdCostRMSE,
                     aveRMSE=aveRMSE, stdRMSE=stdRMSE,
                     ranks=ranks,
                     score=score, total_score=total_score)

            df_result = pd.DataFrame({'aveCostIR': aveCostIR,
                                      'stdCostIR': stdCostIR,
                                      'aveCostRMSE': aveCostRMSE,
                                      'stdCostRMSE': stdCostRMSE,
                                      'aveRMSE': aveRMSE, 'stdRMSE': stdRMSE,
                                      'score': total_score}, index=af_name_list)
            df_result.to_excel(out_path + '%s_Result_%d.xlsx' % (bench_name, cost))

            name = []
            data_name_list = ['costIR', 'costRMSE', 'RMSE', 'Rank', 'Score']
            label_name_list = ['Convergence Cost(IR)',
                               'Convergence Cost(RMSE)',
                               'Convergence RMSE',
                               ]
            for data_name in data_name_list:
                for af in af_name_list:
                    name.append('%s_%s' % (data_name, af))
            df = pd.DataFrame(np.hstack((cAllCostIR, cAllCostRMSE, cAllRMSE, ranks, score)), columns=name)
            df.to_excel(out_path + '%s_Data_%d.xlsx' % (bench_name, cost))

            # 结果可视化
            data_list = [cAllCostIR, cAllCostRMSE, cAllRMSE]
            for l in range(len(data_list)):
                data = []
                for m in range(len(af_name_list)):
                    value = data_list[l][:, m]
                    data.append(value)
                plt.grid(True)  # 显示网格
                plt.boxplot(data,
                            medianprops={'color': 'red', 'linewidth': '1.5'},
                            meanline=True,
                            showmeans=True,
                            meanprops={'color': 'blue', 'ls': '--', 'linewidth': '1.5'},
                            showfliers=True,
                            flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 10},
                            autorange=True,
                            # autorange=False,
                            labels=af_name_list)
                # plt.yticks(np.arange(0.4, 0.81, 0.1))
                # plt.title('%s' % (bench_name))
                plt.ylabel(label_name_list[l])
                plt.savefig(out_path + '%s_%d_%s.png' % (bench_name, cost, data_name_list[l]))
                plt.show()
                plt.close()

    return
