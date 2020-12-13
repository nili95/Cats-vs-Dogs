# -*- coding: utf-8 -*-
"""
@author: NILI
"""


import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import cv2
import os


# Load image

path = 'Z:/inhouse/nili/ML/archive/500_train/'
cat1 = cv2.imread(os.path.join(path,'dogs','dog.222.jpg'),0)
cat1=cat1/255
cat1 = cat1.astype('float32')
rows, cols = np.array(cat1.shape)
print(rows,cols)

# Display image
fig = plt.figure()
ax = fig.add_subplot(111)

ax.imshow(cat1,cmap='gray')
ax.set_axis_off()
plt.show()

# singular values
U,s,VT = np.linalg.svd(cat1,full_matrices=True)

fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111)
ax.plot(np.log10(s),color='blue')
ax.set_xlabel('Sing value index $i$', fontsize=16)
ax.set_ylabel('$\log_{10}(\sigma_i)$', fontsize=16)
ax.set_title('Singular Values', fontsize=18)
plt.show()

# Find and display low-rank approximations

r_vals = np.array([10, 20, 30,40, 50, 60,70,80,90,100])
#r_vals = np.array([10, 20, 30,50,100 ])

err_fro = np.zeros(len(r_vals))

# display images of various rank approximations
fig, axs = plt.subplots(1,5,figsize=(18,4))
for i in range(len(r_vals)):
   
    ind = int(r_vals[i]-1)
    # Complete and uncomment two lines below
    cat1_r = U[:,:ind+1]@(np.diag(s[:ind+1]))@VT[:ind+1,:]
    Er = (cat1-cat1_r)
    err_fro[i] = np.linalg.norm(Er,ord='fro')
    
    #ax = fig.add_subplot(111)
    axs[i].imshow(cat1_r,cmap='gray',interpolation='none')
    ax.set_axis_off()
    axs[i].set_title(['Rank =', str(r_vals[i])], fontsize=18)
    plt.show()
    
# plot normalized error versus rank
norm_err = err_fro/np.linalg.norm(cat1,ord='fro')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.stem(r_vals,norm_err,linefmt='grey', markerfmt='bo')
ax.set_xlabel('Rank', fontsize=16)
ax.set_ylabel('Normalized error', fontsize=16)
plt.show()

# bias-variance tradeoff
num_sv = min(rows, cols)
bias_2 = np.zeros(num_sv)
ranks = np.arange(num_sv)

for r in range(num_sv):
    bias_2[r] = np.linalg.norm(s[r:num_sv])**2

sigma2 = 10
var = sigma2*ranks
#print(var)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(ranks,np.log10(bias_2),'r',linewidth=2,label='Bias squared')
ax.plot(ranks[1:],np.log10(var[1:]),'b', linewidth=2, label = 'Variance')
ax.plot(ranks,np.log10(bias_2+var),'g', linewidth=2, label='Bias squared + Variance')
min_bias_plus_variance_index = np.argmin(np.log10(bias_2+var))
ax.plot(ranks[min_bias_plus_variance_index], np.log10(bias_2+var)[min_bias_plus_variance_index], '*', markersize=15)
ax.set_xlabel('Rank', fontsize=16)
ax.set_ylabel('$\log_{10}$ of error', fontsize=16)
ax.legend()
plt.show()
print(ranks[min_bias_plus_variance_index])
