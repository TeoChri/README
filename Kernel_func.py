# import matplotlib.pyplot as plt
import numpy as np


def R_GAUSS(X1, X2, theta_P):
    '''This functaion calculates the correlation between
    the points X1 and X2 based on the distance function'''
    nth = len(theta_P)
    nTheta = int(nth / 2)
    theta = theta_P[: nTheta].reshape(1, -1)
    P = theta_P[nTheta:].reshape(1, -1)

    n1 = len(X1)
    n2 = len(X2)

    D = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            # for k in range(len(theta)):
            r = abs((X1[i] - X2[j]).reshape(1, -1))
            d = (r ** P) * theta
            D[i, j] = np.sum(d)

    R = np.exp(-D)
    return R

def R_GAUSS6(X1, X2, theta_P):
    '''This functaion calculates the correlation between
    the points X1 and X2 based on the distance function'''

    theta = theta_P[0]
    P = theta_P[1]

    n1 = len(X1)
    n2 = len(X2)

    D = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            # for k in range(len(theta)):
            r = abs((X1[i] - X2[j]).reshape(1, -1))
            d = (r ** P) * theta
            D[i, j] = np.sum(d)

    R = np.exp(-D)
    return R

def R_GAUSS2(X1, X2, theta_P):
    '''This functaion calculates the correlation between
     the points X1 and X2 based on the distance function'''
    theta = theta_P.reshape(1, -1)

    n1 = len(X1)
    n2 = len(X2)

    D = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            # for k in range(len(theta)):
            r = abs((X1[i] - X2[j]).reshape(1, -1))
            d = (r ** 2) * theta
            D[i, j] = np.sum(d)

    R = np.exp(-D)
    return R


def R_GAUSS_PER(X1, X2, theta_P):
    '''This functaion calculates the correlation between
    the points X1 and X2 based on the distance function'''
    nth = len(theta_P)

    theta = theta_P[: nth / 2].reshape(1, -1)
    P = theta_P[(nth) / 2:].reshape(1, -1)

    n1 = len(X1)
    n2 = len(X2)

    D = np.zeros((n1, n2))
    PD = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            # for k in range(len(theta)):
            r = abs((X1[i] - X2[j]).reshape(1, -1))
            d = (r ** P) * theta
            D[i, j] = np.sum(d)

            pd = np.sin(np.pi / r) ** 2 * 2
            PD[i, j] = np / sum(pd)

    R = np.exp(-D) + np.exp(-PD)
    return R


def R_GAUSS_SYM(X1, X2, theta_P):
    '''This functaion calculates the correlation between
    the points X1 and X2 based on the distance function'''
    nth = len(theta_P)

    theta = theta_P[: nth / 2].reshape(1, -1)
    P = theta_P[(nth) / 2:].reshape(1, -1)

    n1 = len(X1)
    n2 = len(X2)

    D = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            # for k in range(len(theta)):
            r = abs((X1[i] - X2[j]).reshape(1, -1))
            r = 0.5 - abs(0.5 - r)
            d = (r ** P) * theta
            D[i, j] = np.sum(d)

    R = np.exp(-D)
    return R


def R_PER(X1, X2, theta_P):
    '''This functaion calculates the correlation between
    the points X1 and X2 based on the distance function'''
    nth = len(theta_P)

    theta = theta_P[: nth / 2].reshape(1, -1)
    P = theta_P[(nth) / 2:].reshape(1, -1)

    n1 = len(X1)
    n2 = len(X2)

    PD = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            # for k in range(len(theta)):
            r = abs((X1[i] - X2[j]).reshape(1, -1))

            pd = np.sin(np.pi / r) ** 2 * 2
            PD[i, j] = np / sum(pd)

    R = np.exp(-PD)
    return R


# def R_Matern32(X1, X2, theta_P):
#     theta = theta_P.reshape(1,-1)
#
#     n1 = len(X1)
#     n2 = len(X2)
#
#     D = np.zeros((n1, n2))
#     R = np.zeros((n1, n2))
#     for i in range(n1):
#         for j in range(n2):
#             # for k in range(len(theta)):
#             r = abs((X1[i] - X2[j]).reshape(1,-1))
#             d = r ** 2 * theta
#             D[i,j] = np/sum(d)
#
#
#     R = np.exp(-D * np.sqrt(3)) * (1 + np.sqrt(3) * D)
#     return R
#
#     r = distance / length_scale
#     res = (1 + np.sqrt(3) * r) * np.exp(- r * np.sqrt(3))

def Rcompute(X1, X2, theta_P, type):
    if type == 1:
        R = R_GAUSS(X1, X2, theta_P)
    elif type == 2:
        R = R_GAUSS_PER(X1, X2, theta_P)
    # elif type == 3:
    #     R = R_GEK(X1, X2, theta_P)
    elif type == 4:
        R = R_GAUSS_SYM(X1, X2, theta_P)
    elif type == 5:
        R = R_PER(X1, X2, theta_P)
    elif type == 11:
        R = R_GAUSS2(X1, X2, theta_P)
    elif type == 6:
        R = R_GAUSS6(X1, X2, theta_P)

    return R

