import numpy as np

def RMSE(y_actual, y_predict):
    actual = y_actual.reshape(-1, 1)
    predict = y_predict.reshape(-1, 1)
    mse = np.mean((actual - predict) ** 2)
    rmse = np.sqrt(mse)

    return rmse

def MAE(y_actual, y_predict):
    actual = y_actual.reshape(-1, 1)
    predict = y_predict.reshape(-1, 1)

    mae = np.mean(abs(actual - predict))
    return mae

def MAPE(y_actual, y_predict):
    actual = y_actual.reshape(-1, 1)
    predict = y_predict.reshape(-1, 1)

    mape = np.mean(abs(actual - predict) / abs(actual))
    return mape

def Rsquare(y_actual, y_predict):
    actual = y_actual.reshape(-1, 1)
    predict = y_predict.reshape(-1, 1)

    RSS = np.sum((actual - predict) ** 2)
    TSS = np.sum((actual - np.mean(actual)) ** 2)
    rsquare = 1 - RSS / TSS
    return rsquare

# 计算Simple Regret
def SR(y_opt, y_train_best):
    sr = y_opt - y_train_best
    return sr

# 计算Inference Regret
def IR(y_opt, f_X_best):
    ir = y_opt - f_X_best
    return ir