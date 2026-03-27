#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 15:34:43 2025

@author: joris
"""

#%% RUN FIRST
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 13:27:04 2024

@author: joris
"""#%% RUN FIRST
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 13:27:04 2024

@author: joris
"""
SERVER=False

datadir='/home/joris/Dropbox/Articles/RampCliff/Data/'

def subscript(ax,i,color='k',bg='w',x=0.03,y=0.93,script=['a)','b)','c)','d)']):
	txt=ax.text(x,y,script[i],color=color,transform = ax.transAxes,backgroundcolor=bg)
	return txt

plt.style.use(datadir+'joris.mplstyle')

from matplotlib.colors import ListedColormap
Fire=np.loadtxt(datadir+'LUT_Fire.csv',delimiter=',',skiprows=1)
Fire=Fire/255.
cm_fire=ListedColormap(Fire[:,1:], name='Fire', N=None)


# figdir=''
# datadir=''

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import ndimage, misc
import numpy.fft
from scipy.ndimage import gaussian_filter
import cv2
import time
import os

from scipy.fft import fft, ifft,fftfreq

def subscript(ax,i,color='k',bg='w',x=0.03,y=0.93,script=['a)','b)','c)','d)']):
	txt=ax.text(x,y,script[i],color=color,transform = ax.transAxes,backgroundcolor=bg)
	return txt

plt.style.use(datadir+'joris.mplstyle')

from matplotlib.colors import ListedColormap
Fire=np.loadtxt(datadir+'LUT_Fire.csv',delimiter=',',skiprows=1)
Fire=Fire/255.
cm_fire=ListedColormap(Fire[:,1:], name='Fire', N=None)


def spectral(Np=int(2**9),p=1,tmax=20,dt=0.5,D=1e-4,seed=12,a=1,SLICE=-1):
	C=np.zeros((Np,Np*p))
	#C[:,:(Np*p)//2]=1.
	
	
	#C[(Np*p)//4:3*(Np*p)//4,(Np*p)//4:3*(Np*p)//4]=1.
	#C=C-np.mean(C)
	
	# save slice
	if SLICE>0:
		Cslice=[]
		
	X,Y=np.meshgrid(np.arange(int(Np*p)),np.arange(Np))
	ky=2*np.pi*np.tile(fftfreq(C.shape[0], d=1.0/C.shape[0]),(C.shape[1],1)).T
	kx=2*np.pi*np.tile(fftfreq(C.shape[1], d=1.0/C.shape[0]),(C.shape[0],1))
	k=np.sqrt(ky**2+kx**2)
	
	#start with a small thin lamella
	C=np.exp(-(X-Np/2.)**2/(2*5**2))
	C[(Y>Np*0.6)|(Y<Np*0.4)]=0
	
	# Start from the fourier transform of concentration field
	fC=fft(fft(C,axis=0),axis=1)
	if SLICE>0:
		C=np.real(ifft(ifft(fC,axis=0),axis=1))
		Cslice.append(C)
	np.random.seed(seed=seed)
	PhaseX=np.random.rand(10000)*2*np.pi
	PhaseY=np.random.rand(10000)*2*np.pi
	Vc=[]
	PSD=[]
	for t in range(tmax):
		print(t)
		vX=a*np.sin(Y/Np*2*np.pi+PhaseX[t])
		vY=a*np.sin(X/Np*2*np.pi+PhaseY[t])
		# Half period
		for t in np.arange(0,0.5,dt):
			fCx = ifft(fC,axis=0)
			dcx=np.exp(1j*kx*vX*dt)*fCx
			fC=fft(dcx,axis=0)*np.exp(-D*k**2*dt)#			# Without source term
		# 2nd Half period
		for t in np.arange(0,0.5,dt):
			fCy = ifft(fC,axis=1)
			fC=fft(np.exp(1j*ky*vY*dt)*fCy,axis=1)*np.exp(-D*k**2*dt)
			if SLICE>0:
				C=np.real(ifft(ifft(fC,axis=0),axis=1))
				Cslice.append(C)
		Vc.append(np.mean(np.abs(fC)**2)/Np**2)
		PSD.append(np.mean(np.abs(fCx)**2,axis=0)[:fCx.shape[0]//2])
	
	if SLICE>0:	
		print('here')
		return np.real(ifft(ifft(fC,axis=0),axis=1)),np.array(Vc),np.array(PSD),kx[0,:kx.shape[0]//2],np.array(Cslice)
	else:			
		return np.real(ifft(ifft(fC,axis=0),axis=1)),np.array(Vc),np.array(PSD),kx[0,:kx.shape[0]//2]
	
#%% Run simulation
Np=int(2**9)
a=0.5 # amplitude of sine wave
D=1e-6 #diffusion


#Theoretical lyapunov (Kraichhnan)
lyap=a**2*np.pi**2/8


C,Vc,PSD,kx,Cslice=spectral(Np=Np,a=a,tmax=int(6/lyap),D=D,SLICE=1)


plt.figure()
plt.imshow(Cslice[0,:,:])

Cm=np.mean(Cslice)


plt.figure()
plt.imshow(Cslice[0,:,:]+np.random.randn(Cslice.shape[1],Cslice.shape[2])*0.01)

plt.figure()
plt.imshow(Cslice[0,:,:]+np.random.randn(Cslice.shape[1],Cslice.shape[2])*0.1)

#%% Effect of background noise on decay of scalar variance
plt.figure()
for noise in [1e-2,1e-3,1e-4]:
	Cflat=Cslice.reshape(Cslice.shape[0],-1)
	Cflat=Cflat+np.random.randn(Cflat.shape[0],Cflat.shape[1])*noise
	# Simple variance of each time snapshot
	V=np.var(Cflat,axis=1)
	plt.plot(V,label='noise={:1.1e}'.format(noise))
	plt.yscale('log')
plt.legend()

t=np.arange(len(V))
tfit=np.uint16([0.5/lyap,3/lyap])
p=np.polyfit(t[tfit[0]:tfit[1]],np.log(V)[tfit[0]:tfit[1]],1)

plt.plot(t,np.exp(p[1]+p[0]*t),'k--',label=r'$\exp(-{:1.2f}t)$'.format(-p[0]))
plt.xlabel(r'time')
plt.ylabel(r'$\sigma^2_c$')
plt.legend()

#%% Mean over threshold

#Theoretical lyapunov (Kraichhnan)
lyap=a**2*np.pi**2/8


#Parameters to play with
tfit=[2,5]
noise=0.01
#tfit=np.uint16([0.5/lyap,3/lyap])
#print(tfit)


Cflat=Cslice.reshape(Cslice.shape[0],-1)


Eps=[0,1e-4,5e-4,1e-3,5e-3,1e-2,2e-2,3e-2,5e-2,1e-1]
Eps=np.logspace(-2.5,-0.7,20)

plt.figure()
alpha_n=[]
gamma_n=[]
for i,eps in enumerate(Eps):
	n=2
	Ctemp=np.copy(Cflat)+np.random.randn(Cflat.shape[0],Cflat.shape[1])*noise
	Ctemp[Ctemp<eps]=np.nan
	V=np.nanmean(Ctemp**n,axis=1)
	Var=np.var(Cflat,axis=1)
	plt.plot(V/V[0],color=plt.cm.jet(i/len(Eps)))
	t=np.arange(len(V))
	alpha_n.append(-np.polyfit(t[tfit[0]:tfit[1]],np.log(V)[tfit[0]:tfit[1]],1)[0])
	gamma_n.append(-np.polyfit(t[tfit[0]:tfit[1]],np.log(Var)[tfit[0]:tfit[1]],1)[0])

plt.plot(Var/Var[0],'k--',linewidth=1.5,label=r'$\sigma^2_c$')
plt.yscale('log')
plt.xlabel(r'time')
plt.ylabel(r'$<c^2|c>\epsilon>$')
plt.legend()

plt.figure()
plt.plot(Eps,alpha_n,'*',label=r'$\alpha_2$')
plt.plot(Eps,gamma_n,'o',label=r'$\gamma_2$')
plt.plot(Eps,np.zeros(len(Eps))+lyap/2,'k:',label=r'$\lambda/2 $')
plt.plot(Eps,np.zeros(len(Eps))+lyap,'k--',label=r'$\lambda$ ')
plt.plot(Eps,np.zeros(len(Eps))+2*lyap,'k-',label=r'$2\lambda$ ')
plt.xlabel(r'$\epsilon$')
plt.ylabel(r'$\alpha_{2,\epsilon}$')
plt.ylim([0,2.5*lyap])
plt.xscale('log')
plt.legend()

t=tfit[1]
plt.figure()
plt.imshow(Cslice[t,:,:]+np.random.randn(Cslice.shape[1],Cslice.shape[2])*noise>Eps[0])

plt.figure()
plt.imshow(Cslice[t,:,:]+np.random.randn(Cslice.shape[1],Cslice.shape[2])*noise>Eps[len(Eps)//2])

plt.figure()
plt.imshow(Cslice[t,:,:]+np.random.randn(Cslice.shape[1],Cslice.shape[2])*noise>Eps[-1])
