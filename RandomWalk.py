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
        if i % d == 0:
            X[i//d] += 1
        else:
            X[i//d] -= 1
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
            print(n, i, X0)
            if n >= N:
                break
            elif k >= 10*d:
                break
            X1 = np.copy(X0)
            if i % d == 0:
                X1[i//d] += 1
            else:
                X1[i//d] -= 1
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

if __name__ == "__main__":
    '''
    Self-avoiding random walk on a square lattice
    '''
    plt.ioff()
    plt.close('all')


    #simulating diffusion in d dimensions
    d = 2 #number of dimensions
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

    # XX2 = []
    # for n in range(L):
    #     X2, _ = BrownianWalk(d, N)
    #     XX2.append(X2)

    # XX2 = np.array(XX2)

    # fig, ax = plt.subplots(1, 1, figsize = (12, 6))
    # ax.plot(np.arange(N), XX2[:5, :].T, ls = '--')
    # ax.plot(np.arange(N), XX2.mean(axis=0), label = r'\langle X^2\rangle$')
    # ax.set_xlabel('n')
    # ax.set_ylabel(r'$X^2, \langle X^2\rangle$')
    # plt.savefig('./figs/X2_of_n.jpg')
    # plt.close()

    # fig, axs = plt.subplots(1, 2, figsize = (12, 6))
    # axs[0].plot(np.arange(N), XX2.mean(axis=0), label = r'\langle X^2\rangle$')
    # axs[0].set_xlabel('n')
    # axs[0].set_ylabel(r'$\langle X^2\rangle$')

    # axs[1].loglog(np.arange(N), XX2.mean(axis=0), label = r'\langle X^2\rangle$')
    # axs[1].set_xlabel('n')
    # axs[1].set_ylabel(r'$\langle X^2\rangle$')
    # plt.savefig('./figs/X2_mean.jpg')
    # plt.close()

    if d == 2:
        fig, axs = plt.subplots(1, 2, figsize = (12, 6))
        X2, XX = BrownianWalk(d, N)
        for (x1, y1), (x2, y2) in zip(XX[:-1], XX[1:]):
            axs[0].plot([x1, x2], [y1, y2], color = 'b')
        axs[0].set_xlabel('x')
        axs[0].set_ylabel('y')

        X2, XX = SelfAvoidingWalk(d, N)
        for (x1, y1), (x2, y2) in zip(XX[:-1], XX[1:]):
            axs[1].plot([x1, x2], [y1, y2], color = 'b')
        axs[1].set_xlabel('x')
        axs[1].set_ylabel('y')
        plt.savefig('./figs/traj.jpg')
        plt.close()


