#!/usr/bin/env python

from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
import sys, os
import seaborn as sns
sns.set()


np.random.seed()

def BrownianWalk(d, N):
    '''
    Simulating Brownian motion in d dimensions.
    Arguments:
    d : number of dimensions
    N : number of steps

    Returns:
    XX : trajectory
    X2 : squared distance from th eorigin
    '''
    X = np.zeros(d)
    ii = np.random.randint(0, 2*d, N)
    X2 = []
    XX = [tuple(X)]
    for n, i in enumerate(ii):
        if i % 2 == 0:
            X[i//2] += 1
        else:
            X[i//2] -= 1
        X2.append(X.dot(X))
        XX.append(tuple(X))

    return X2, XX

def SelfAvoidingWalk(d, N):
    '''
    Simulating self-avoiding random walk in d dimensions.
    Arguments:
    d : number of dimensions
    N : number of steps

    Returns:
    XX : trajectory
    X2 : squared distance from th eorigin
    '''
    X0 = np.zeros(d)
    X2 = []
    XX = [tuple(X0)]
    n = 0
    k = 0
    while n < N and k < 10*d:
        ii = np.random.randint(0, 2*d, 10**6)
        for i in ii:
            if n >= N:
                break
            elif k >= 10*d:
                break
            X1 = np.copy(X0)
            if i % 2 == 0:
                X1[i//2] += 1
            else:
                X1[i//2] -= 1
            if tuple(X1) in XX:
                k += 1
                continue
            else:
                X2.append(X1.dot(X1))
                XX.append(tuple(X1))
                X0 = np.array(X1)
                n += 1
                k = 0
                # print(n, X1)

    return X2, XX

def padding(XX, N):
    '''
    pad sublists shorter than N with zeros and return an array
    '''
    YY = []
    for X in XX:
        if len(X) < N:
            YY.append(np.append(np.array(X), np.zeros(N - len(X))))
        else:
            YY.append(X)
    return np.array(YY)

if __name__ == "__main__":
    '''
    Self-avoiding random walk on a square lattice
    '''
    plt.ioff()
    plt.close('all')


    #simulating diffusion in d dimensions
    d = 4 #number of dimensions
    N = 10**3 #number of steps
    L = 10**2 #size of the ensemble (number of realizations)

    # X = np.zeros(d)
    # ii = np.random.randint(0, 2*d, N)
    # X2 = []
    # for n, i in enumerate(ii):
    #     if i % d == 0:
    #         X[i//d] += 1
    #     else:
    #         X[i//d] -= 1
    #     X2.append(X.dot(X))
    #     print(X)

    # X2, XX = BrownianWalk(d, N)
    # sys.exit()

    XX2 = []
    YY2 = []
    dd = [2,3,4,5]
    for d in dd:
        XX2_d = []
        YY2_d = []
        for n in range(L):
            X2, _ = BrownianWalk(d, N)
            XX2_d.append(X2)
            Y2, _ = SelfAvoidingWalk(d, N)
            YY2_d.append(Y2)
        XX2_d = np.array(XX2_d)
        XX2.append(XX2_d)
        YY2.append(YY2_d)

    YY2pad = [padding(YY2_d, N) for YY2_d in YY2]
    Y2_mean = [YY2_d.sum(axis=0)/(YY2_d>0).sum(axis=0) for YY2_d in YY2pad]

    fig, axs = plt.subplots(2, 1, figsize = (8, 8))
    axs[0].plot(np.arange(N), XX2[0][:5, :].T, ls = '--')
    axs[0].plot(np.arange(N), XX2[0].mean(axis=0), label = r'$\langle X^2\rangle$')
    axs[0].set_xlabel('n')
    axs[0].set_ylabel(r'$X^2, \langle X^2\rangle$')
    axs[0].set_title('Diffusion')
    axs[0].legend()

    for Y2 in YY2[0][:5]:
        axs[1].plot(np.arange(len(Y2)), Y2, ls = '--')
    axs[1].plot(np.arange(N), Y2_mean[0], label = r'$\langle X^2\rangle$')
    axs[1].set_xlabel('n')
    axs[1].set_ylabel(r'$X^2, \langle X^2\rangle$')
    axs[1].set_title('Self-avoiding walk')
    axs[1].legend()
    plt.savefig('./figs/X2_of_n.jpg')
    plt.close()

    fig, axs = plt.subplots(2, 2, figsize = (8, 8))
    for d, XX2_d, Y2_mean_d in zip(dd, XX2, Y2_mean):
        axs[0, 0].plot(np.arange(N), XX2_d.mean(axis=0), label = f'd = {d}')
        axs[0, 1].loglog(np.arange(N), XX2_d.mean(axis=0), label = f'd = {d}')
        axs[1, 0].plot(np.arange(N), Y2_mean_d, label = f'd = {d}')
        axs[1, 1].loglog(np.arange(N), Y2_mean_d, label = f'd = {d}')
    for ax in axs:
        ax[0].set_xlabel('n')
        ax[0].set_ylabel(r'$\langle X^2\rangle$')
        ax[0].legend()
        ax[0].set_title('Diffusion')

        ax[1].set_xlabel('n')
        ax[1].set_ylabel(r'$\langle X^2\rangle$')
        ax[1].legend()
        ax[1].set_title('Self-avoiding walk')

    plt.savefig('./figs/X2_mean.jpg')
    plt.close()

    fig, axs = plt.subplots(1, 2, figsize = (12, 6))
    X2, XX = BrownianWalk(2, N)
    for (x1, y1), (x2, y2) in zip(XX[:-1], XX[1:]):
        axs[0].plot([x1, x2], [y1, y2], color = 'b')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')

    X2, XX = SelfAvoidingWalk(2, N)
    for (x1, y1), (x2, y2) in zip(XX[:-1], XX[1:]):
        axs[1].plot([x1, x2], [y1, y2], color = 'b')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')
    plt.savefig('./figs/traj.jpg')
    plt.close()


