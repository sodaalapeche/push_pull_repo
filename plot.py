# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 11:01:48 2024

@author: battais
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

code from leont


"""
# comments data intresting : 1523

import numpy as np
import matplotlib.pyplot as plt
import csv
import matplotlib.image as mpimg
import skimage as ski

from scipy.ndimage import gaussian_filter,median_filter
import skimage

#%% folder 1 Leon

folder='D:/Leon/ppull_SABLE_homog_22_11_3/'
file=folder+'ppull_SABLE_homog_22_11_3_MMStack_Default.ome.tif'
dt=2.5


I=skimage.io.imread(file)
Iz=np.float32(I.reshape(I.shape[0],-1))/2**16

T=np.arange(I.shape[0])*dt


#%% folder 2 Kostis

folder='D:/Kostis/btc_SABLE_3dp_02_12_4/'
file=folder+'btc_SABLE_3dp_02_12_4_MMStack_Default.ome.tif'
dt=3


I=skimage.io.imread(file)
Iz=np.float32(I.reshape(I.shape[0],-1))/2**16

T=np.arange(I.shape[0])*dt





def first_pos(liste):
    for i,arg in enumerate(liste):
        if arg == max(liste):
            return i
        
    return -1


#%% 
A = Iz-Iz[0,:]

Q_A = 50                  #ml/min    
dt_A = 3                 #s
lc_A =0.25                 #m big bead  ... see if I want to change it
#%%


Q2_A=(Q_A*(10**(-6)))/60      #m3/s
S=(0.057/2)**2*np.pi    #m2
porosite=0.5            # peut =-être à mesurer pour les impressions 3D  intérssannt
umean_A= Q2_A/(S*porosite)  #m/s
ta_A=lc_A/umean_A             #referent time in s ; temps d'advetion


#####     data       #####

tta_A=T/ta_A         #time from the data in s

                      # find on the graph
# idx1=np.where((tta>0.5)&(tta<0.8))[0]

# p1 = np.polyfit(np.log(tta[idx1]), np.log(stdmean[idx1]), 1)



# Pexp1 = []
# for i in range (len(stdmean)):
#     Pexp1.append(tta[i]**p1[0] * np.exp(p1[1]))

variable=np.var(A,axis=1)/np.mean(A,axis=1)**2
idx0=np.where((tta_A>0.3)&(tta_A<1.02))[0]    
p= np.polyfit(np.log(tta_A[idx0]), np.log(variable[idx0]), 1)
Pexp = []
for i in range (len(tta_A)):
    Pexp.append(tta_A[i]**p[0] * np.exp(p[1]))


plt.figure()
plt.plot(tta_A,variable,label='variance/meansquared')
# plt.plot(tta_A, 1/tta_A, label= '1/t')
# plt.plot(tta_A, 1/tta_A**2, label= '1/t**2')
# plt.plot(tta_A, 1/tta_A**(3/2), label= '1/t**3/2')
# plt.plot(tta_A, Pexp, label= "a="+f"{p[0]:.2f}" )
plt.yscale('log')
plt.xscale('log')
plt.xlabel('time')
plt.ylabel('$\sigma^2$')
plt.legend()



variable=np.mean(A,axis=1)
plt.figure()
plt.plot(tta_A,variable,label='mean')
# plt.plot(tta_A,1e10*T**(-2.30),label='man')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('time ta')
plt.ylabel('$\mu$')
# plt.plot(np.var(Iz,axis=1),label='Variance')
# plt.yscale('log')

# plt.plot(np.mean(Iz,axis=1)**2,label='Mean squared')
# plt.yscale('log')
# plt.xlim([0.1,5])
# plt.ylim([1,10000])

plt.legend()
#%% data sans only 2
#A=np.loadtxt(r"C:/Users/leont/Desktop/Stage_Rennes_2024/py_stage_rennes_2024/share/Experiences/homogeneous/Push_pull/3D8printing/push_pull-3D_PRINTED_08-09_3/Results.csv", delimiter=',',skiprows=1)
A1 = np.loadtxt(r"D:/Kostis/btc_SABLE_3dp_02_12_4/Results.csv", delimiter=',',skiprows=1)
#♣A=np.loadtxt(r"C:/Users/leont/Desktop/Stage_Rennes_2024/py_stage_rennes_2024/share/Experiences/homogeneous/Push_pull/Joris/Results3.csv", delimiter=',',skiprows=1)


Q_A1 = 50                  #ml/min    
dt_A1 = 5                 #s
lc_A1 =0.35                 #m big bead  ... see if I want to change it


# do not change

Q2_A1=(Q_A1*(10**(-6)))/60      #m3/s
S=(0.057/2)**2*np.pi    #m2
porosite=0.5            # peut =-être à mesurer pour les impressions 3D
umean_A1= Q2_A1/(S*porosite)  #m/s
ta_A1=lc_A1/umean_A1             #referent time in s ; temps d'advetion


#####     data       #####

t_A1=A1[:,0]* dt_A1            #time from the data in s
std_A1=A1[:,2]              #standard deviation, directely from the data?
mean_A1=A1[:,1]             #same?
tta_A1=t_A1/ta_A1                #dimensionless time
stdmean_A1=std_A1/mean_A1        # standart deviation normalised



#%%"""fitting"""
fp_A1= first_pos(stdmean_A1)

#dimension

# tta_A1 = tta_A1 - tta_A1[fp_A1 -5]

stdmean_A1=stdmean_A1/stdmean_A1[fp_A1 +10]

    
#%%""plot"""
# t = np.linspace(0,1,num= np.size(stdmean_B[fp_B:]))





plt.figure(dpi=200)
plt.plot(tta_A1[fp_A1-5:], stdmean_A1[fp_A1-5:], markerfacecolor='none',color='black', markersize=1, label='sand only2')



# plt.plot(tta_C, np.exp(-0.3 * tta_C*17+0.2), label= 'exp-0.3')
# plt.plot(tta[idx0[0]:idx0[-1]], Pexp[idx0[0]:idx0[-1]], color='blue', label= "a="+f"{p[0]:.2f}")
# plt.plot(tta[idx1[0]-3:idx1[-1]+5], Pexp1[idx1[0]-3:idx1[-1]+5], label= "a="+f"{p1[0]:.2f}")

# plt.xscale('log')
plt.yscale('log')
plt.legend(loc='lower left', prop = {'size': 7} )
plt.grid()
plt.title('Push Pull log-log comparison with and without beads')
plt.xlabel('$t/t_a$')
plt.ylabel('$\sigma /\mu$')
# plt.xlim([0, 0.4])
plt.show()


# plt.figure(dpi=200)

# plt.plot(tta, stdmean, markerfacecolor='none', color='red', markersize=1, label='StdDev/Mean')
# #plt.plot(tta[idx1[0]-10:idx1[-1]+10], Pexp2[idx1[0]-10:idx1[-1]+10],label= "a="+f"{p2[0]:.2f}")
# #plt.plot(tta, np.exp(-*tta))

# plt.yscale('log')
# plt.legend(loc='upper right')
# plt.grid()
# plt.title('Push Pull log-normal')
# plt.xlabel('$t/t_a$')
# plt.ylabel('StdDev/Mean')
# plt.xlim([tta[fp-50],500/ta])
# plt.show()





#%%"""plot mean"""





plt.figure(dpi=200)
plt.plot(tta_A1, mean_A1, markerfacecolor='none', color='red', markersize=1)#, label='Mean')
# plt.plot(tta[idx0[0]-10:idx0[-1]+10], Pexp[idx0[0]-10:idx0[-1]+10], color='blue', label= "a="+f"{p[0]:.2f}")
# plt.plot(tta[idx1[0]-10:idx1[-1]+10], Pexp1[idx1[0]-10:idx1[-1]+10], label= "a="+f"{p1[0]:.2f}")

plt.xscale('log')
plt.yscale('log')
plt.legend(loc='upper right')
plt.grid()
# plt.yscale('log')
# plt.xscale('log')
plt.title('Push Pull log-log')
plt.xlabel('$t/t_a$')
plt.ylabel('Mean')
#plt.xlim([tta[fp],500/ta])
plt.show()















