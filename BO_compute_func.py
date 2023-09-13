import numpy as np
from scipy.stats import norm
from sklearn.utils.extmath import cartesian


# import time
# from tqdm import tqdm
#
# for i in tqdm(range(100), desc='Progress', ncols=100, ascii=' =', bar_format='{l_bar}{bar}|'):
#     time.sleep(0.05)


# def Opt_Random_Sample(
#         mean,
#         cov,
#         Size: int = 10,
#         RNG=None,
#         maximize: bool = True):
#     y_best = np.zeros((Size, 1))
#
#     y_sample = RNG.multivariate_normal(np.squeeze(mean), cov, size=Size)
#     if maximize:
#         for i in range(Size):
#             y_best[i, 0] = np.max(y_sample[i, :])
#     else:
#         for i in range(Size):
#             y_best[i, 0] = np.min(y_sample[i, :])
#
#     return y_best

def Opt_Random_Sample(
        model,
        input_dim: int,
        grid_size: int = None,
        Size: int = 10,
        RNG=None,
        maximize: bool = True):
    # if grid_size is None:
    #     if input_dim == 1:
    #         grid_size = 1000
    #     elif input_dim == 2:
    #         grid_size = 50
    #     elif input_dim == 3:
    #         grid_size = 20
    #     elif input_dim==4:
    #         grid_size = 6
    #     else:
    #         grid_size = 2
    #
    # grid_1d = np.linspace(0, 1, grid_size)
    # x_pool = cartesian([grid_1d for _ in range(input_dim)])
    # mean = model.meanPredict(x_pool)
    # cov = model.varPredict(x_pool, cov=True)  # self.__
    #

    if RNG is None:
        RNG = np.random.default_rng(0)
    if input_dim == 1:
        sample_size = 100
    elif input_dim == 2:
        sample_size = 400
    else:
        sample_size = 1000
    x_pool = RNG.random((sample_size, input_dim))
    mean = model.meanPredict(x_pool)
    cov = model.varPredict(x_pool, cov=True)  # self.__

    y_best = np.zeros((Size, 1))

    y_sample = RNG.multivariate_normal(np.squeeze(mean), cov, size=Size)
    if maximize:
        for i in range(Size):
            y_best[i, 0] = np.max(y_sample[i, :])
    else:
        for i in range(Size):
            y_best[i, 0] = np.min(y_sample[i, :])

    return y_best


def mes_compute(y_best_set,
                mean,
                std,
                maximize: bool = True,
                ):
    mean = np.asarray(mean).reshape(-1, 1)
    std = np.asarray(std).reshape(-1, 1)

    if not maximize:
        mean = -mean
        y_best_set = -y_best_set

    # 对std进行截断
    std_t = np.zeros(std.shape)
    scale = 0.45
    std_p = np.log(std) * scale
    for i in range(len(std)):
        std_t[i] = 0.1 if std[i] < 0.1 else std[i]
        std_p[i] = scale * np.log(1e-2) if std_p[i] < (scale * np.log(1e-2)) else std_p[i]

    gamma = (y_best_set.T - mean - std_p) / std_t #- np.log(std)
    # gamma = (y_best_set.T - mean - np.log(std)) / std
    # gamma = (y_best_set.T - mean ) / std_t

    gamma_cdf = norm.cdf(gamma)
    gamma_pdf = norm.pdf(gamma)

    # for i in range(len(gamma)):
    #     for j in range(len(y_best_set)):
    #         if gamma_pdf[i, j] < 1e-10:
    #             gamma_pdf[i, j] = 1e-10

    res = gamma * gamma_pdf / (2 * gamma_cdf) - np.log(gamma_cdf)

    # Check the invalid value.
    for i in range(len(mean)):
        for j in range(len(y_best_set)):
            if res[i, j] >= 0:
                pass
            else:
                res[i, j] = 0
            if gamma_cdf[i, j] == 0:
                res[i, j] = 0


    # Average for the sampling times of y_best_set.
    af_value = np.mean(res, axis=1)
    return af_value


def inf_mes_compute(y_best_set,
                    mean_h,
                    std_h,
                    std_l,
                    rho,
                    # cov_l_h,
                    maximize: bool,
                    l_bnd: float = -3,
                    up_bnd: float = 3,
                    num: int = 200,
                    ):
    # var = std_h ** 2 - cov_l_h**2 / (std_l**2)
    var = std_h ** 2 - rho ** 2 * std_l ** 2
    std = np.zeros((len(var), 1))
    for j in range(len(var)):
        std[j] = np.sqrt(var[j]) if var[j] > 0 else std_h[j]

    res = 0
    scale = np.linspace(l_bnd, up_bnd, num)
    for delta in scale:
        mean = mean_h + delta * rho

        value = mes_compute(y_best_set,
                            mean=mean,
                            std=std,
                            maximize=maximize)

        res += value * norm.pdf(delta) * (up_bnd - l_bnd) / num

    return res


def dmes_compute(y_best_set,
                 mean_h,
                 std_h,
                 std_l,
                 rho,
                 inf_num: int = 200,
                 cost: float = 1,
                 maximize: bool = True):
    mean_h = mean_h.reshape(-1, 1)
    std_h = std_h.reshape(-1, 1)
    std_l = std_l.reshape(-1, 1)

    '''Compute the MES for HF'''
    mes_h = mes_compute(y_best_set,
                        mean=mean_h,
                        std=std_h,
                        maximize=maximize)

    res_h = mes_h / cost  # /3

    '''Compute the MES for HF conditioned on the LF samples. '''
    # var_star = std_h ** 2 - rho ** 2 * std_l ** 2
    # var_star = std_h ** 2 - cov_l_h ** 2 / (std_l ** 2)
    # mes_h_on_l = inf_mes_compute(y_best_set,
    #                              mean_h,
    #                              std_h,
    #                              std_l,
    #                              rho,
    #                              # cov_l_h,
    #                              maximize=maximize,
    #                              l_bnd=-3,
    #                              up_bnd=3,
    #                              num=inf_num)
    var = std_h ** 2 - rho ** 2 * std_l ** 2
    std = np.zeros((len(var), 1))
    for j in range(len(var)):
        # std[j] = np.sqrt(var[j]) if var[j] > 0 else 0
        std[j] = np.sqrt(var[j]) if var[j] > 0 else std_h[j]

    mes_h_on_l = mes_compute(y_best_set,
                             mean=mean_h,
                             std=std,
                             # cov_l_h,
                             maximize=maximize, )

    res_l = mes_h - mes_h_on_l
    for i in range(len(res_l)):
        # if var[i] <= 0:
        #     res_l[i] = 0
        if res_l[i] < 0:
            res_l[i] = 0
        if std_l[i] == 0:
            res_l[i] = 0

    return res_h, res_l


def ei_compute(mean,
               std,
               y_best,
               x0: float = 0.1,
               maximize: bool = True):
    size = len(mean)
    af_value = np.zeros(size)
    for i in range(size):
        if std[i] != 0:
            if maximize:
                z = (mean[i] - y_best - x0) / std[i]
            else:
                z = (y_best - mean[i] - x0) / std[i]
            af_value[i] = std[i] * (z * norm.cdf(z) + norm.pdf(z))
        else:
            af_value[i] = 0

        if af_value[i] >= 0:
            pass
        else:
            af_value[i] = 0

    return af_value


def inf_ei_compute(y_best,
                   mean_h,
                   std_h,
                   std_l,
                   rho,
                   maximize: bool,
                   l_bnd: float = -3,
                   up_bnd: float = 3,
                   num: int = 200,
                   ):
    var = std_h ** 2 - rho ** 2 * std_l ** 2
    std = np.zeros((len(var), 1))
    for j in range(len(var)):
        std[j] = np.sqrt(var[j]) if var[j] > 0 else std_h[j]

    res = 0
    scale = np.linspace(l_bnd, up_bnd, num)
    for delta in scale:
        mean = mean_h + delta * rho

        value = ei_compute(y_best=y_best,
                           mean=mean,
                           std=std,
                           maximize=maximize)

        res += value * norm.pdf(delta) * (up_bnd - l_bnd) / num
    return res


def efi_compute(y_best,
                mean_h,
                std_h,
                std_l,
                rho,
                cost: float = 1,
                maximize: bool = True):
    mean_h = mean_h.reshape(-1, 1)
    std_h = std_h.reshape(-1, 1)
    std_l = std_l.reshape(-1, 1)

    ei_h = ei_compute(mean=mean_h,
                      std=std_h,
                      y_best=y_best,
                      maximize=maximize)

    res_h = ei_h / cost  # /3

    '''Compute the MES for HF conditioned on the LF samples. '''
    var_star = std_h ** 2 - rho ** 2 * std_l ** 2
    ei_h_on_l = inf_ei_compute(y_best,
                               mean_h,
                               std_h,
                               std_l,
                               rho,
                               maximize=maximize,
                               l_bnd=-3,
                               up_bnd=3,
                               num=200)

    res_l = ei_h - ei_h_on_l
    for i in range(len(res_l)):
        if var_star[i] <= 0:
            res_l[i] = 0
        if res_l[i] < 0:
            res_l[i] = 0
        if std_l[i] == 0:
            res_l[i] = 0
    return res_h, res_l
