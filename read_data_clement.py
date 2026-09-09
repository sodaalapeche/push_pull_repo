#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 15:51:33 2026

@author: joris
"""



from scipy.ndimage import gaussian_filter,median_filter
import numpy as np
import matplotlib.pyplot as plt
import re
DATA=np.load('/home/chorus/Downloads/data2(3).npy',allow_pickle=True)


#%%
GOOD=np.arange(len(DATA))
GOOD=np.array([0, 1,2, 3, 4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 16, 17])
L=30

colorBY='Pe'# 'Pe','Inclusion'

cmap=plt.cm.inferno

exptype=['hetero']

alldata=[]

for j,i in enumerate(GOOD):
	label=DATA[i]['label']
	d2= int(re.findall(r'\d+', label)[0])
	if 'fine' in label:
		d1=0.08
	if 'coarse' in label:
		d1=0.6

	to_consider = [e in label for e in exptype]
	if np.all(to_consider):
		print(label)
		Pe=d2/d1
		t=DATA[i]['time']
		idt0=np.where(t>0)[0][0]
		ta=DATA[i]['Ta']
		x=DATA[i]['time']#/DATA[i]['Ta']
		var_norm=DATA[i]['var']/DATA[i]['mean']**2
		mean=DATA[i]['mean']

		# FIND TA !
		gradmean=gaussian_filter(np.diff(DATA[i]['mean'])/np.diff(t),3)
		n=len(gradmean)//2
		idmax=np.where(gradmean[:n]==np.max(gradmean[:n]))[0][0]
		idmin=np.where(gradmean[n:]==np.min(gradmean[n:]))[0][0]+n
		ta=(t[idmin]-t[idmax])/L


		# Data to plot
		gradmean=gaussian_filter(np.diff(DATA[i]['mean'])/np.diff(t/ta),3)
		# grad Mean
		#data= mean
		# grad Mean
		#data=np.abs(gradmean)
		#data=np.abs(gradmean)/np.mean(mean[idmax:idmin])
		#alldata.append([d1,d2,data[idmin]])
		#Variance
		#data=var_norm/np.nanmax(var_norm)
		data=var_norm/var_norm[idt0]
		# Slope
# 		data=-gaussian_filter(np.diff(np.log(var_norm))/np.diff(t/ta),1)
# 		idmid=(idmax+idmin)//2
# 		alldata.append([d1,d2,np.mean(data[int(0.8*idmid):int(1.1*idmid)])])


		id0=np.where(x>0)[0][0]
		D=1e-2
		t0 = 1/(D*data[id0])
		#plt.plot(x+t0,data)
		if colorBY == 'sand':
			color=cmap(d1)
	#		lab='$d_1={:1.1f}$'.format()
		if colorBY == 'Pe':
			color=cmap(np.log10(Pe)/2)
		if colorBY == 'Inclusion':
			color=cmap(d2/10)

		#plt.plot((t[:len(data)]-t[idmax])/ta,data,color=color)
		#plt.plot((t[:len(data)])/ta,data,color=color)
		#plt.plot((t[:])/ta,data,color=plt.cm.jet(j/len(GOOD)),label='{:d}'.format(i))


#plt.legend()
plt.yscale('log')
#plt.xscale('log')
plt.xlim([-1,L*1.2])
#plt.ylim([-10,10])
plt.show()

#%%
R=0.025
Pe=np.linspace(1,100,10)
A=np.array(alldata)
plt.scatter(A[:,1]/A[:,0],A[:,-1]*R**2/0.01/(A[:,0]*1e-3) ,s=10*A[:,0])
plt.plot(Pe,Pe,'k--',label='$Pe$')
plt.plot(Pe,Pe**0.5,'k-',label='$Pe^{0.5}$')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Pe')
plt.ylabel(r'$D_M/D_m \sim \alpha_T / d_1 \sim - \beta R^2 / (\ell d_1) $')
plt.legend()

#%%
A=np.array(alldata)

plt.scatter(A[:,1]/A[:,0],(1/A[:,-1])**2/0.30*0.01**2,s=10*A[:,0])
plt.xlabel('$Pe = d_2/d_1$')
plt.ylabel(r'$\alpha_\parallel $ [m]')
plt.yscale('log')
plt.xscale('log')
