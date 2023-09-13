import pandas as pd
import numpy as np

path = 'C:\\Users\\Teo\\Desktop\\0504\\NACA0012\\'


def expNACA(i_x, fidelity: str = 'HF', i_loc=0):
    # path = 'C:\\Users\\Teo\\Desktop\\0504\\NACA0012'
    history_filename = []
    for k in range(len(i_x)):
        idx_name = str(i_x[k]).zfill(3)
        filename = path + 'n%sm' % (idx_name)
        history_filename.append(filename)

    data = readHistoryData(historyFilename=history_filename, nameData='2', i_loc=i_loc)
    if fidelity=='LF':
        data = HF2LF(data)
    return data


def expAllNACA(fidelity: str = 'HF',nameData='2', i_loc=0):
    # path = 'C:\\Users\\Teo\\Desktop\\1\\grid_size%d\\'%grid_size

    history_filename = []
    for k in range(0, 120):
        idx_name = str(k).zfill(3)
        filename = path + 'n%sm' % (idx_name)
        history_filename.append(filename)

    data = readHistoryData(historyFilename=history_filename, nameData=nameData, i_loc=i_loc)
    if fidelity=='LF':
        data = HF2LF(data)
    return data


def readHistoryData(historyFilename, nameData='2', i_loc=0):
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
        data_of_interest = hist_data[nameData].values[ i_loc]
        data.append(data_of_interest)
    return np.asarray(data).reshape(num_history, 1) * 10000


def dat2csv():
    num_history = 120
    for i_hist in range(num_history):

        datalist = []
        with open(path + 'n%sm.dat' % str(i_hist+1).zfill(3), 'r') as of:
            for line in of:
                # 清除前后回车符，按照空格进行分割
                line = line.strip('\n').split()
                linelist = []
                for string in line:
                    # 按照‘,’进行分割
                    # 字符串前后可能还有','，需要先清除
                    linelist = linelist + (string.split(' '))
                linelist = list(map(float, linelist))
                datalist.append(linelist)
        df = pd.DataFrame(datalist)
        df.to_csv(path + 'n%sm.csv' % str(i_hist).zfill(3))
    return

def HF2LF(data):
    res = 0.8 * data + np.sin(data)
    return res