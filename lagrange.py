#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 23 16:32:04 2025

@author: joris
"""
import numpy as np
import matplotlib.pyplot as plt

filename = "imper"
U = np.load('/data1/clement/' + filename + '/ux.npy')
V = np.load('/data1/clement/' + filename + '/uy.npy')
W = np.load('/data1/clement/' + filename + '/uz.npy')
cas = "inclu"
if cas == 'homogene':
    U = np.ones(np.shape(U))
    V = np.zeros(np.shape(V))
    W = np.zeros(np.shape(W))
# Since velocity is defined on faces, take centered
u = U[:, :-1, :]
v = V[:-1, :, :]
w = W[:, :, :-1]

# %% Lagrangian trajectories on eulerian velocity field
from scipy.interpolate import RegularGridInterpolator

# =============================================================================
# You need to supply
ux, uy, uz = U, V, W
#
# =============================================================================
SAVETRAJ = True

dtraj = 50

np.random.seed(10)
dx = 1 / 10.

y = np.arange(ux.shape[0])
x = np.arange(ux.shape[1] - 1)
z = np.arange(ux.shape[2])

U = np.array([ux[:, 1:, :], uy[1:, :, :], uz[:, :, 1:]])

Unorm = np.sum(U ** 2, axis=3) ** 0.5

fx = RegularGridInterpolator((y * dx, x * dx, z * dx), uy[1:, :, :], bounds_error=False)
fy = RegularGridInterpolator((y * dx, x * dx, z * dx), ux[:, 1:, :], bounds_error=False)
fz = RegularGridInterpolator((y * dx, x * dx, z * dx), uz[:, :, 1:], bounds_error=False)


def f(L):  # interpolator
    return np.array([fx(L), fy(L), fz(L)]).T


it = 130000

npart = 50

# random
L = np.array([np.random.rand(npart) * 2., np.zeros(npart) + dx / 2., np.random.rand(npart) * 2.]).T

# Grid

xp, yp = np.meshgrid(np.linspace(0.01, 0.9, npart) * ux.shape[0] * dx, np.linspace(0.01, 0.9, npart) * ux.shape[2] * dx)
L = np.vstack((xp.flatten(), np.zeros(npart * npart) + dx, yp.flatten())).T

L0 = np.copy(L)

cfl = 0.9
dt = cfl * dx / np.max(Unorm)

FPOS = np.linspace(0, ux.shape[1] - 4, 10) * dx

Lend = [np.copy(L) for f in range(len(FPOS))]
Tend = [np.zeros(L.shape[0]) for f in range(len(FPOS))]
Lall = [np.copy(L)[np.arange(0, L.shape[0], dtraj), :]]
idgood = len(L)
i = 0

while (i < it) & (idgood > 0):
    idgood = np.nansum(L[:, 1] < FPOS[-1])
    print(i, idgood)
    k1 = f(L)
    k2 = f(L + dt / 2 * k1)
    k3 = f(L + dt / 2 * k2)
    k4 = f(L + dt * k3)
    Lold = np.copy(L)
    L += dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    # L+=dt*k1
    # print(L)
    # keep position on transverse planes
    for k, fpos in enumerate(FPOS):
        # print(fpos,np.sum(idin),np.nanmax(L[:,1]))
        idin = (L[:, 1] >= fpos) * (Lold[:, 1] < fpos)
        Lend[k][idin, :] = L[idin, :]
        Tend[k][idin] = i * dt

    if SAVETRAJ:
        Lall.append(np.copy(L)[np.arange(0, L.shape[0], dtraj), :])
    i += 1

if SAVETRAJ:
    Lall = np.array(Lall)

    plt.figure(figsize=(5, 5))
    [plt.plot(Lall[:, i, 2].T, Lall[:, i, 0].T, '-', color=plt.cm.jet(i / Lall.shape[1])) for i in range(Lall.shape[1])]

    plt.show()

    plt.figure()
    [plt.plot(Lall[:, i, 1].T, Lall[:, i, 0].T, '-', color=plt.cm.jet(i / Lall.shape[1])) for i in range(Lall.shape[1])]
    plt.show()

    plt.figure()
    [plt.plot(Lall[:, i, 1].T, Lall[:, i, 2].T, '-', color=plt.cm.jet(i / Lall.shape[1])) for i in range(Lall.shape[1])]

    plt.show()
fig, ax = plt.subplots(2, 5, figsize=(10, 4))
[ax.flatten()[i].scatter(Lend[i][:, 2], Lend[i][:, 0], color=plt.cm.jet(np.arange(L0.shape[0]) / L0.shape[0]), s=0.1)
 for i in range(len(Lend))]
[ax.flatten()[i].axis('off') for i in range(len(Lend))]

plt.show()
# fig,ax=plt.subplots(2,5,figsize=(5,2))
# [ax.flatten()[i].imshow(np.log(kperm[:,int(FPOS[i]/dx),:])) for i in range(len(Lend))]
# [ax.flatten()[i].axis('off') for i in range(len(Lend))]


# for i,t in enumerate(Tend):
#	h,x=np.histogram(t,np.logspace(1,3,100),density=True)
#	plt.plot(x[1:],h,'+',color=plt.cm.jet(i/len(Tend)))
# plt.yscale('log')
# plt.xscale('log')
# plt.xlabel('arrival time')
# plt.ylabel('probability')