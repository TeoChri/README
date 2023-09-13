import chaospy as cp
import numpy as np
from sklearn.linear_model import LarsCV


def PCE(xTrain, yTrain, lb=None, ub=None,):
    dim = int(xTrain.size / len(yTrain))

    # distri = []
    # for i in range(dim):
    #     if lb is not None:
    #         distri.append(cp.Uniform(lb[i], ub[i]))
    #     else:
    #         distri.append(cp.Uniform(0, 1))

    # joint = cp.J(term for term in distri)
    joint = cp.Uniform(0, 1)

    expansion = cp.generate_expansion(9, joint, normed=True)
    lars = LarsCV(fit_intercept=False, max_iter=5)
    pce, coeffs = cp.fit_regression(expansion,
                                    xTrain.T,
                                    np.squeeze(yTrain),
                                    model=lars,
                                    retall=True)
    expansion_ = expansion[coeffs != 0]

    approx_solver = cp.fit_regression(expansion_,
                                      xTrain.T,
                                      yTrain,)
    #
    def f(x):
        x = x.reshape(-1, dim)
        res = approx_solver(x.T).reshape(len(x), 1)
        return res

    return f

    # return approx_solver