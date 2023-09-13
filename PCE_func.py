import numpy as np
import openturns as ot


def PCE(xTrain, yTrain, totalDegree=7):
    dim = int(xTrain.size / len(xTrain))
    xTrain = xTrain.reshape(-1, dim)
    yTrain = np.asarray(yTrain).reshape(-1, 1)

    xTrain = ot.Sample(xTrain)
    yTrain = ot.Sample(yTrain)
    input_dimension = xTrain.getDimension()

    distri = []
    for i in range(input_dimension):
        distri.append(ot.Uniform())
    distribution = ot.ComposedDistribution(distri)

    # construct the multi variate basis
    polyColl = ot.PolynomialFamilyCollection(input_dimension)
    for i in range(input_dimension):
        marginal = distribution.getMarginal(i)
        polyColl[i] = ot.StandardDistributionPolynomialFactory(marginal)

    enumfunc = ot.LinearEnumerateFunction(input_dimension)
    indexMax = enumfunc.getStrataCumulatedCardinal(totalDegree)

    multivariateBasis = ot.OrthogonalProductPolynomialFactory(polyColl, enumfunc)

    # set the adaptive strategy
    adaptive_strategy = ot.FixedStrategy(multivariateBasis, indexMax)
    # projection_strategy = ot.LARS() #ot.LeastSquaresStrategy()
    # projection_strategy = ot.LeastSquaresStrategy(xTrain,
    #                                               yTrain,
    #                                               ot.LARS())
                                                  # ot.LeastSquaresMetaModelSelectionFactory) #(,ot.CorrectedLeaveOneOut())
    approximationAlgorithm = ot.LeastSquaresMetaModelSelectionFactory(ot.LARS(), ot.CorrectedLeaveOneOut())
    projection_strategy = ot.LeastSquaresStrategy(approximationAlgorithm)
    # projection_strategy = ot.LeastSquaresStrategy()


    # generate pce model
    pce = ot.FunctionalChaosAlgorithm(xTrain,
                                      yTrain,
                                      distribution,
                                      adaptive_strategy,
                                      projection_strategy)
    pce.run()
    result_pce = pce.getResult()

    # coe = result_pce.getCoefficients()
    # print('PCE 的系数为：')
    # print(coe)
    def f(x):
        x = x.reshape(-1, dim)
        res = result_pce.getMetaModel()(x)
        res = np.asarray(res).reshape(len(x), -1)
        return res

    return f
