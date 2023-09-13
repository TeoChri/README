import numpy as np
import numpy.random as random
import copy
############## InputDimension = 1 ##################
'''
    A standard gaussian noise with 10^(-2) scale is added to the function.
    The transformed input bounds are 0 <= xi <= 1, i = 1.
    :param X: array type, shape = (n,1),n>=1.
    :param find_min: bool type, since the acquisition functions are targeted to find the maxmum,
                    if the glabal mimimum is more valuebale, set find_min=false(0); else set find_min=ture(1).
    :return: array type, shape = (n,1),n>=1.
'''
eps = 1e-5 #10 ** (-3)

'''The Forrester function.(Dim=1)'''


# .. math::
#     f_h(x) = (6x-2)^2 \sin(12x-4)
# .. math::
#     f_l(x) = A f_h(x) + B(x-0.5) + C
# With :math:`A=0.5, B=10` and :math:`C=-5` as recommended parameters.
#  The initial input bounds are 0 <= xi <= 1, i = 1.x_opt = [0.757248757841856]
#  The minimum value is      without noise.
def FO(X, eps=eps):
    dim = 1
    X = X.reshape(-1, dim)

    x1 = X

    term1 = (6 * x1 - 2) ** 2
    term2 = np.sin(12 * x1 - 4)

    y = term1 * term2
    y = y + eps * random.randn(y.shape[0], 1)
    return y


def FOa_low(X, eps=eps, A=0.5, B=10, C=-5):
    dim = 1
    X = X.reshape(-1, dim)

    x1 = X
    term1 = (6 * x1 - 2) ** 2
    term2 = np.sin(12 * x1 - 4)
    term3 = B * (x1 - 0.5)
    term4 = C

    y = A * term1 * term2 + term3 + term4
    y = y + eps * random.randn(y.shape[0], 1)
    return y

def FOb_low(X, eps=eps, A=1, B=0, C=-5):
    dim = 1
    X = X.reshape(-1, dim)

    x1 = X
    term1 = (6 * x1 - 2) ** 2
    term2 = np.sin(12 * x1 - 4)
    term3 = B * (x1 - 0.5)
    term4 = C

    y = A * term1 * term2 + term3 + term4
    y = y + eps * random.randn(y.shape[0], 1)
    return y

def FOc_low(X, eps=eps, A=1, B=0, C=0):
    dim = 1
    X = X.reshape(-1, dim)

    x1 = X + 0.2
    term1 = (6 * x1 - 2) ** 2
    term2 = np.sin(12 * x1 - 4)
    term3 = B * (x1 - 0.5)
    term4 = C

    y = A * term1 * term2 + term3 + term4
    y = y + eps * random.randn(y.shape[0], 1)
    return y

'''The Sasena function.(Dim=1)'''


#  The global minimums are at () . The minimum value is      without noise.
#  The initial input bounds are 0 <= xi <= 1, i = 1.
def Sasena(X, eps=eps):
    dim = 1
    X = X.reshape(-1, dim)

    x = X * 10

    y = - np.exp(x/100) - np.sin(x) + 10
    y = y + eps * random.randn(y.shape[0], 1)
    return y


def Sasena_low(X, eps=eps):
    dim = 1
    X = X.reshape(-1, dim)

    x = X * 10

    y = - np.exp(x/100) - np.sin(x) + 10.3 + 0.03 * (x-3)**2
    y = y + eps * random.randn(y.shape[0], 1)
    return y


'''The One Dimejsion function.(Dim=1)'''


#  The global minimums are at () . The minimum value is      without noise.
#  The initial input bounds are 0 <= xi <= 1, i = 1.
def OD(X, eps=eps):
    dim = 1
    X = X.reshape(-1, dim)

    x = X

    y = ((6 * x - 2) ** 2 * np.sin(12 * x - 4))
    y = y + eps * random.randn(y.shape[0], 1)

    return y


def OD_low(X, eps=eps):
    dim = 1
    X = X.reshape(-1, dim)

    x = X

    y_original = ((6 * x - 2) ** 2 * np.sin(12 * x - 4))
    y = 0.5 * y_original + 10 * (x - 0.5) - 5
    y = y + eps * random.randn(y.shape[0], 1)

    return y


'''The Gramacy-Lee function.(Dim=1)'''


# The global minimums are at () .The minimum value is      without noise.
# The initial input bounds are 0.5 <= x1 <= 2.5.
def GL(X, eps=eps):
    dim = 1
    X = X.reshape(-1, dim)

    x1 = X

    x1bar = 1 * (2 * x1 + 0.5)
    term1 = np.sin(10 * np.pi * x1bar) / (2 * x1bar)
    term2 = (x1bar - 1) ** 4

    y = term1 + term2
    y = y + eps * random.randn(y.shape[0], 1)
    return y


def GL_low(X, eps=eps):
    dim = 1
    X = X.reshape(-1, dim)

    x1 = X

    x1bar = 0.7 * (2 * x1 + 0.5)
    term1 = np.sin(10 * np.pi * x1bar) / (2 * x1bar)
    term2 = (x1bar - 1) ** 4

    y = term1 + term2 * 0.5
    y = y + eps * random.randn(y.shape[0], 1)
    return y


############## InputDimension = 2 ##################
''' The Currin function.(Dim=2)'''


# .. math::
#     f_h(x_1, x_2) = \Bigg( 1 - \exp(-\dfrac{1}{2x_2})\Bigg) \dfrac{2300x_1^3 +
#                     1900x_1^2 + 2092x_1 + 60}{100x_1^3 + 500x_1^2 + 4x_1 + 20}
# .. math::
#     f_l(x_1, x_2) = (&f_h(x_1+0.05, x_2+0.05) + \\
#                      &f_h(x_1+0.05, x_2-0.05) + \\
#                      &f_h(x_1-0.05, x_2+0.05) + \\
#                      &f_h(x_1-0.05, x_2-0.05)) / 4
# Here we add a standard gaussian noise with 10^(-3) scale to the function.
# The input bounds are 0 <= xi <= 1, i = 1,2.x_opt = [0.21666666666666, 0]
# The expected bounds are 0 <= xi <= 1, i = 1,2.x_opt = [0.21666666666666, 0]
def f_currin(x1, x2):
    x1 = x1.reshape(-1, 1)
    x2 = x2.reshape(-1, 1)

    are_zero = x2 <= 1e-8  # Assumes x2 approaches 0 from positive
    fact1 = np.ones(x2.shape)

    fact1[~are_zero] -= np.exp(-1 / (2 * x2[~are_zero]))
    fact2 = 2300 * (x1 ** 3) + 1900 * (x1 ** 2) + 2092 * x1 + 60
    fact3 = 100 * (x1 ** 3) + 500 * (x1 ** 2) + 4 * x1 + 20

    y = fact1 * fact2 / fact3
    return y


def Currin(X, eps=eps):
    dim = 2
    X = X.reshape(-1, dim)

    x1 = X[:, 0].reshape(-1, 1)
    x2 = X[:, 1].reshape(-1, 1)

    y = f_currin(x1, x2)
    y = y + eps * random.randn(y.shape[0], 1)
    return y


def Currin_low(X, eps=eps):
    dim = 2
    X = X.reshape(-1, dim)

    x1 = X[:, 0].reshape(-1, 1)
    x2 = X[:, 1].reshape(-1, 1)

    x1_plus = (x1 + .05).reshape(-1, 1)
    x1_minus = (x1 - .05).reshape(-1, 1)
    x2_plus = (x2 + .05).reshape(-1, 1)
    x2_minus = (x2 - .05).reshape(-1, 1)
    x2_minus[x2_minus < 0] = 0

    yh1 = f_currin(x1_plus, x2_plus)
    yh2 = f_currin(x1_plus, x2_minus)
    yh3 = f_currin(x1_minus, x2_plus)
    yh4 = f_currin(x1_minus, x2_minus)

    y = (yh1 + yh2 + yh3 + yh4) / 4
    y = y + eps * random.randn(y.shape[0], 1)

    return y


'''The Bukin  function.(Dim=2)'''


# The global minimums are at (-10,1) .
# The minimum value is      without noise.
# Here we add a standard gaussian noise with 10^(-3) scale to the function.
# The input bounds are 0 <= xi <= 1, i = 1,2.
# The expected bounds are -15 <= x1 <= -5, -3 <= x1 <= 3.
def Bukin(X, eps=eps):
    dim = 2
    X = X.reshape(-1, dim)

    x1 = X[:, 0].reshape(-1, 1)
    x2 = X[:, 1].reshape(-1, 1)

    x1bar = 1 * (20 * x1 - 15)
    x2bar = 1 * (6 * x2 - 3)
    term1 = 100 * np.sqrt(abs(x2bar - 0.01 * x1bar ** 2))
    term2 = 0.01 * abs(x1bar + 10)

    y = term1 + term2
    y = y + eps * random.randn(y.shape[0], 1)

    return y


def Bukin_low(X, eps=eps):
    dim = 2
    X = X.reshape(-1, dim)

    x1 = X[:, 0].reshape(-1, 1)
    x2 = X[:, 1].reshape(-1, 1)

    x1bar = 1 * (20 * x1 - 15)
    x2bar = 1 * (6 * x2 - 3)
    term1 = 100 * np.sqrt(abs(x2bar - 0.01 * x1bar ** 2))
    term2 = 0.01 * abs(x1bar + 10)

    x1star = 1 * (20 * x1 - 15)
    x2star = 1 * (6 * x2 - 3)
    term3 = x1star * x2star - 15

    y = term1 + term2 + term3
    y = y + eps * random.randn(y.shape[0], 1)

    return y


'''The Branin function.(Dim=2)'''


# .. math::
#     f_b(x_1, x_2) = \Bigg(x_2 - (5.1\dfrac{x_1^2}{4\pi^2}) + \dfrac{5x_1}{\pi} -
#                     6\Bigg)^2 + \Bigg(10\cos(x_1) (1 - \dfrac{1}{8\pi}\Bigg) + 10
# .. math::
#     f_h(x_1, x_2) = f_b(x_1, x_2) - 22.5x_2
# .. math::
#     f_l(x_1, x_2) = f_b(0.7x_1, 0.7x_2) - 15.75x_2 + 20(0.9 + x_1)^2 - 50
# The global minimums are at (0.12389382, 0.81833333),(0.54277284,0.15166667) and (0.961652, 0.165).
# The minimum value is -0.39788735780357776 without noise.
# Here we add a standard gaussian noise with 10^(-3) scale to the function.
# The input bounds are 0 <= xi <= 1, i = 1,2.
# The expected bounds are -5 <= x1 <= 10, 0 <= x2 <= 15. x_opt = [-3.786088705282203, 15]
def Branin(X, eps=eps):
    dim = 2
    X = X.reshape(-1, dim)

    x1 = X[:, 0].reshape(-1, 1)
    x2 = X[:, 1].reshape(-1, 1)

    x1bar = 15 * x1 - 5
    x2bar = 15 * x2

    term1 = x2bar - 5.1 * x1bar ** 2 / (4 * np.pi ** 2) + 5 * x1bar / np.pi - 6
    term2 = (10 - 10 / (8 * np.pi)) * np.cos(x1bar)

    y = (term1 ** 2 + term2 + 10) - 22.5 * x2bar
    y = y + + eps * random.randn(y.shape[0], 1)

    return y


def Branin_low(X, eps=eps):
    dim = 2
    X = X.reshape(-1, dim)

    x1 = X[:, 0].reshape(-1, 1)
    x2 = X[:, 1].reshape(-1, 1)

    x1bar = 0.7 * (15 * x1 - 5)
    x2bar = 0.7 * (15 * x2)
    term1 = x2bar - 5.1 * x1bar ** 2 / (4 * np.pi ** 2) + 5 * x1bar / np.pi - 6
    term2 = (10 - 10 / (8 * np.pi)) * np.cos(x1bar)
    y_orignal = (term1 ** 2 + term2 + 10)

    x1star = 15 * x1 - 5
    x2star = 15 * x2
    term3 = 20 * (0.9 + x1star) ** 2
    term4 = 15.75 * x2star

    y = y_orignal + term3 - term4 - 50
    # term3 = 2 * (x1star - 0.5)
    # term4 = -3 * (3 * x2star - 1)
    #
    # y = np.sqrt(y_orignal) * 10 + term3 - term4
    y = y + eps * random.randn(y.shape[0], 1)

    return y


'''The Six-hump camel-back function.(Dim=2)'''


# .. math::
#     f_h(x_1, x_2) = 4x_1^2 - 2.1x_1^4 + \dfrac{x_1^6}{3} + x_1x_2 - 4x_2^2 + 4x_2^4
# .. math::
#     f_l(x_1, x_2) = f_h(0.7x_1, 0.7x_2) + x_1x_2 - 15
# The minimum value is      without noise.
# Here we add a standard gaussian noise with 10^(-3) scale to the function.
# The input bounds are 0 <= xi <= 1, i = 1,2.
# The expected bounds are -2 <= xi <= 2, i = 1,2.x_opt = [0.0898, -0.7126]
def SC(X, eps=eps):
    dim = 2
    X = X.reshape(-1, dim)

    x1 = X[:, 0].reshape(-1, 1)
    x2 = X[:, 1].reshape(-1, 1)

    x1bar = 1 * (4 * x1 - 2)
    x2bar = 1 * (4 * x2 - 2)
    term1 = 4 * x1bar ** 2 - 2.1 * x1bar ** 4 + x1bar ** 6 / 3
    term2 = x1bar * x2bar - 4 * x2bar ** 2 + 4 * x2bar ** 4

    y = term1 + term2
    y = y + eps * random.randn(y.shape[0], 1)

    return y


def SC_low(X, eps=eps):
    dim = 2
    X = X.reshape(-1, dim)

    x1 = X[:, 0].reshape(-1, 1)
    x2 = X[:, 1].reshape(-1, 1)

    x1bar = 0.7 * (4 * x1 - 2)
    x2bar = 0.7 * (4 * x2 - 2)
    term1 = 4 * x1bar ** 2 - 2.1 * x1bar ** 4 + x1bar ** 6 / 3
    term2 = x1bar * x2bar - 4 * x2bar ** 2 + 4 * x2bar ** 4

    x1star = 4 * x1 - 2
    x2star = 4 * x2 - 2
    term3 = x1star * x2star - 15

    y = term1 + term2 + term3
    y = y + eps * random.randn(y.shape[0], 1)

    return y


'''The Booth  function.(Dim=2)'''


# .. math::
#     f_h(x_1, x_2) = (x_1 + 2x_2 - 7)^2 + (2x_1 + x_2 - 5)^2
# .. math::
#     f_l(x_1, x_2) = f_h(0.4x_1, x_2) + 1.7x_1x_2 - x_1 + 2x_2
# The minimum value is   without noise.
# Here we add a standard gaussian noise with 10^(-3) scale to the function.
# The input bounds are 0 <= xi <= 1, i = 1,2. x_opt = [0.55, 1.65]
# The expected bounds are -10 <= xi <= 10, i = 1,2. x_opt = [1, 3]
def Booth(X, eps=eps):
    dim = 2
    X = X.reshape(-1, dim)

    x1 = X[:, 0].reshape(-1, 1)
    x2 = X[:, 1].reshape(-1, 1)

    x1bar = 1 * (20 * x1 - 10)
    x2bar = 1 * (20 * x2 - 10)
    term1 = (x1bar + 2 * x2bar - 7) ** 2
    term2 = (2 * x1bar + x2bar - 5) ** 2

    y = term1 + term2
    y = y + eps * random.randn(y.shape[0], 1)

    return y


def Booth_low(X, eps=eps):
    dim = 2
    X = X.reshape(-1, dim)

    x1 = X[:, 0].reshape(-1, 1)
    x2 = X[:, 1].reshape(-1, 1)

    x1bar = 0.4 * (20 * x1 - 10)
    x2bar = 1 * (20 * x2 - 10)
    term1 = (x1bar + 2 * x2bar - 7) ** 2
    term2 = (2 * x1bar + x2bar - 5) ** 2

    x1star = 20 * x1 - 10
    x2star = 20 * x2 - 10
    term3 = - 1.7 * x1star * x2star - x1star + 2 * x2star

    y = term1 + term2 + term3
    y = y + eps * random.randn(y.shape[0], 1)

    return y


'''The Bohachevsky  function.(Dim=2)'''


# .. math::
#     f_h(x_1, x_2) = x_1^2 + 2x_2^2 - 0.3\cos(3\pi x_1) - 0.4\cos(4\pi x_2) + 0.7
# .. math::
#     f_l(x_1, x_2) = f_h(0.7x_1, x_2) + x_1x_2 - 12
# The global minimums are at (0.5, 0.5) .
# The minimum value is      without noise.
# Here we add a standard gaussian noise with 10^(-3) scale to the function.
# The input bounds are 0 <= xi <= 1, i = 1,2. x_opt = [0.5, 0.5]
# The expected bounds are -5 <= xi <= 5, i = 1,2. x_opt = [0, 0]

def Bohachevsky(X, eps=eps):
    dim = 2
    X = X.reshape(-1, dim)

    x1 = X[:, 0].reshape(-1, 1)
    x2 = X[:, 1].reshape(-1, 1)

    x1bar = 1 * (10 * x1 - 5)
    x2bar = 1 * (10 * x2 - 5)
    term1 = x1bar ** 2 - 0.3 * np.cos(3 * np.pi * x1bar)
    term2 = 2 * x2bar ** 2 - 0.4 * np.cos(4 * np.pi * x2bar) + 0.7

    y = term1 + term2
    y = y + eps * random.randn(y.shape[0], 1)

    return y


def Bohachevsky_low(X, eps=eps):
    dim = 2
    X = X.reshape(-1, dim)

    x1 = X[:, 0].reshape(-1, 1)
    x2 = X[:, 1].reshape(-1, 1)

    x1bar = 0.7 * (10 * x1 - 5)
    x2bar = 1 * (10 * x2 - 5)
    term1 = x1bar ** 2 - 0.3 * np.cos(3 * np.pi * x1bar)
    term2 = 2 * x2bar ** 2 - 0.4 * np.cos(4 * np.pi * x2bar) + 0.7

    x1star = 10 * x1 - 5
    x2star = 10 * x2 - 5
    term3 = x1star * x2star - 12

    y = term1 + term2 + term3
    y = y + eps * random.randn(y.shape[0], 1)

    return y


'''The PARK  function.(Dim=4)'''


# .. math::
#     f_h(x_1, x_2) = x_1^2 + 2x_2^2 - 0.3\cos(3\pi x_1) - 0.4\cos(4\pi x_2) + 0.7
# .. math::
#     f_l(x_1, x_2) = f_h(0.7x_1, x_2) + x_1x_2 - 12
# The global minimums are at () .
# The minimum value is      without noise.
# Here we add a standard gaussian noise with 10^(-3) scale to the function.
# The input bounds are 0 <= xi <= 1, i = 1,2. x_opt = []
# The expected bounds are 0 <= xi <= 1, i = 1,2,3,4. x_opt =

def Park(x, eps=eps):
    dim = 4
    X = copy.deepcopy(x).reshape(-1, dim)

    x1 = X[:, 0].reshape(-1, 1)
    x2 = X[:, 1].reshape(-1, 1)
    x3 = X[:, 2].reshape(-1, 1)
    x4 = X[:, 3].reshape(-1, 1)

    for i in range(len(x1)):
        if x1[i] == 0:
            x1[i] = 1e-7

    term1a = x1 / 2
    term1b = np.sqrt(1 + (x2 + x3 ** 2) * x4 / x1 ** 2) - 1
    term1 = term1a * term1b

    term2a = x1 + 3 * x4
    term2b = np.exp(1 + np.sin(x3))
    term2 = term2a * term2b

    y = term1 + term2
    y = y + eps * random.randn(y.shape[0], 1)
    return y


def Park_low(x, eps=eps):
    dim = 4
    X = copy.deepcopy(x).reshape(-1, dim)

    x1 = X[:, 0].reshape(-1, 1)
    x2 = X[:, 1].reshape(-1, 1)
    x3 = X[:, 2].reshape(-1, 1)

    yh = Park(X, eps=0)

    term1 = (1 + np.sin(x1) / 10) * yh
    term2 = -2 * x1 + x2 ** 2 + x3 ** 2

    y = term1 + term2 + 0.5
    y = y + eps * random.randn(y.shape[0], 1)
    return y


'''The PARK2  function.(Dim=4)'''


# .. math::
#     f_h(x_1, x_2) = x_1^2 + 2x_2^2 - 0.3\cos(3\pi x_1) - 0.4\cos(4\pi x_2) + 0.7
# .. math::
#     f_l(x_1, x_2) = f_h(0.7x_1, x_2) + x_1x_2 - 12
# The global minimums are at () .
# The minimum value is      without noise.
# Here we add a standard gaussian noise with 10^(-3) scale to the function.
# The input bounds are 0 <= xi <= 1, i = 1,2. x_opt = []
# The expected bounds are 0 <= xi <= 1, i = 1,2,3,4. x_opt =

def Park2(X, eps=eps):
    dim = 4
    X = X.reshape(-1, dim)

    x1 = X[:, 0].reshape(-1, 1)
    x2 = X[:, 1].reshape(-1, 1)
    x3 = X[:, 2].reshape(-1, 1)
    x4 = X[:, 3].reshape(-1, 1)

    term1 = (2 / 3) * np.exp(x1 + x2)
    term2 = -x4 * np.sin(x3)
    term3 = x3

    y = term1 + term2 + term3
    y = y + eps * random.randn(y.shape[0], 1)
    return y


def Park2_low(X, eps=eps):
    yh = Park2(X, eps=0)

    y = 1.2 * yh - 1
    y = y + eps * random.randn(y.shape[0], 1)
    return y


'''The  Borehole  function.(Dim=8)'''
# The Borehole function models water flow through a borehole.
# Its simplicity and quick evaluation makes it a commonly used function
#   for testing a wide variety of methods in computer experiments.
# The response is water flow rate, in m3/yr.
# The input variables and their usual input ranges are:
#     rw ∈ [0.05, 0.15]	radius of borehole (m)
#     r ∈ [100, 50000]	radius of influence (m)
#     Tu ∈ [63070, 115600]   	transmissivity of upper aquifer (m2/yr)
#     Hu ∈ [990, 1110]	potentiometric head of upper aquifer (m)
#     Tl ∈ [63.1, 116]	transmissivity of lower aquifer (m2/yr)
#     Hl ∈ [700, 820]	potentiometric head of lower aquifer (m)
#     L ∈ [1120, 1680]	length of borehole (m)
#     Kw ∈ [9855, 12045]	hydraulic conductivity of borehole (m/yr)

# For the purposes of uncertainty quantification,
#   the distributions of the input random variables are:
#     rw ~ N(μ=0.10, σ=0.0161812)
#     r ~ Lognormal(μ=7.71, σ=1.0056)
#     Tu ~ Uniform[63070, 115600]
#     Hu ~ Uniform[990, 1110]
#     Tl ~ Uniform[63.1, 116]
#     Hl ~ Uniform[700, 820]
#     L ~ Uniform[1120, 1680]
#     Kw ~ Uniform[9855, 12045]

# .. math::
#     f_h(x_1, x_2) = x_1^2 + 2x_2^2 - 0.3\cos(3\pi x_1) - 0.4\cos(4\pi x_2) + 0.7
# .. math::
#     f_l(x_1, x_2) = f_h(0.7x_1, x_2) + x_1x_2 - 12
# The global minimums are at () .
# The minimum value is      without noise.
# Here we add a standard gaussian noise with 10^(-3) scale to the function.
# The input bounds are 0 <= xi <= 1, i = 1,2. x_opt = []
# The expected bounds are 0 <= xi <= 1, i = 1,2,3,4. x_opt =

def Borehole(X, eps=eps):
    dim = 8
    X = X.reshape(-1, dim)

    rw = X[:, 0].reshape(-1, 1)
    r = X[:, 1].reshape(-1, 1)
    Tu = X[:, 2].reshape(-1, 1)
    Hu = X[:, 3].reshape(-1, 1)
    Tl = X[:, 4].reshape(-1, 1)
    Hl = X[:, 5].reshape(-1, 1)
    L = X[:, 6].reshape(-1, 1)
    Kw = X[:, 7].reshape(-1, 1)

    frac1 = 2 * np.pi * Tu * (Hu - Hl)

    frac2a = 2 * L * Tu / (np.log(r / rw) * rw**2 * Kw)
    frac2b = Tu / Tl
    frac2 = np.log(r / rw) * (1 + frac2a + frac2b)

    y = frac1 / frac2
    y = y + eps * random.randn(y.shape[0], 1)
    return y


def Borehole_low(X, eps=eps):
    dim = 8
    X = X.reshape(-1, dim)

    rw = X[:, 0].reshape(-1, 1)
    r = X[:, 1].reshape(-1, 1)
    Tu = X[:, 2].reshape(-1, 1)
    Hu = X[:, 3].reshape(-1, 1)
    Tl = X[:, 4].reshape(-1, 1)
    Hl = X[:, 5].reshape(-1, 1)
    L = X[:, 6].reshape(-1, 1)
    Kw = X[:, 7].reshape(-1, 1)

    frac1 = 5 * Tu * (Hu - Hl)

    frac2a = 2 * L * Tu / (np.log(r / rw) * rw**2 * Kw)
    frac2b = Tu / Tl
    frac2 = np.log(r / rw) * (1.5 + frac2a + frac2b)

    y = frac1 / frac2
    y = y + eps * random.randn(y.shape[0], 1)
    return y
