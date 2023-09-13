import pandas as pd
import numpy as np
def expResult(x, fidelity: str = 'HF', grid_size=20):
    path = 'C:\\Users\\Teo\\Desktop\\1\\grid_size%d\\'%grid_size

    x0_grid = pd.read_csv(path+'exp_HF_0.csv', skipinitialspace=True)
    x0_grid.columns = x0_grid.columns.str.strip()
    x0_grid = x0_grid['0'].values

    x1_grid = pd.read_csv(path+'exp_HF_1.csv', skipinitialspace=True)
    x1_grid.columns = x1_grid.columns.str.strip()
    x1_grid = x1_grid['0'].values

    x = x.reshape(-1, 2)
    history_filename = []
    for k in range(len(x)):
        i, j = 0, 0
        while x0_grid[i] != x[k, 0]:
            i += 1
        while x1_grid[j] != x[k, 1]:
            j += 1

        idx = i * grid_size + j
        idx_name = str(idx).zfill(4)
        filename = path +'history_exp_%s_%s' % (fidelity, idx_name)
        history_filename.append(filename)

    data = readHistoryData(historyFilename=history_filename, nameData='CL')
    return data

def expAllResult(fidelity: str = 'HF', grid_size=20):
    path = 'C:\\Users\\Teo\\Desktop\\1\\grid_size%d\\'%grid_size

    history_filename = []
    for k in range(grid_size**2):
        idx_name = str(k).zfill(4)
        filename = path +'history_exp_%s_%s' % (fidelity, idx_name)
        history_filename.append(filename)

    data = readHistoryData(historyFilename=history_filename, nameData='CL')
    return data

def readHistoryData(historyFilename, nameData='CD'):
    """
    读取history件并输出需要的数据
    :param historyFilename: surface flow的文件名
    :param nameData: 总共18列数据，选取哪一列作为关注数据
    :return:
        感兴趣的数据，由于history数据是中间的计算过程，因此只返回最后一个数值
    """
    num_history = len(historyFilename)
    data = []
    for i_hist in range(num_history):
        hist_filename = historyFilename[i_hist]
        hist_data = pd.read_csv(hist_filename + '.csv', skipinitialspace=True)
        hist_data.columns = hist_data.columns.str.strip()
        data_of_interest = hist_data[nameData].values[-1]
        data.append(data_of_interest)
    return np.asarray(data).reshape(num_history, 1)