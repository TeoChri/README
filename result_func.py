from plotResult_func import plotResult
import numpy as np

# result_dir = 'C:\\Users\\Teo\\Desktop\\BO Result08\\10'
result_dir = 'C:\\Users\\Teo\\Desktop\\BO Result17\\%d_%d\\' % (10, 0)
bench_name_list = [
    # 'Currin',
    # 'Bukin',
    # 'Branin',
    # 'Six-hump camel-back',
    # 'Booth',
    # 'Bohachevsky',
    # 'Gramacy-Lee',
    'Forrester 1a',
    'Forrester 1b',
    'Forrester 1c',
    'Sasena',
    # 'Park',
    # 'Park2',
    # 'Borehole',
]

af_name_list = [
    'DMES',
    'VFEI',
    'EFI',
    'MES',
    'EI',
]  #
seed_size = 1
seeds = np.arange(seed_size)
# seeds = [1,2,3,4]
measure_list = ['SR',
                'IR',
                'RMSE',
                'R_square']

plotResult(path=result_dir,
           bench_name_list=bench_name_list,
           af_name_list=af_name_list,
           seed_list=seeds,
           measure_list=measure_list)
