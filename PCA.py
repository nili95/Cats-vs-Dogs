# -*- coding: utf-8 -*-
"""
@author: NILI
"""
import numpy as np
import matplotlib.pyplot as plt 
import cv2
import os

def PCA(X):
   
    
    # get matrix dimensions
    num_data, dim = X.shape
    
    # center data
    mean_X = X.mean(axis=0)
    X = X - mean_X
    
    if dim > num_data:
        # PCA compact trick
        M = np.dot(X, X.T) # covariance matrix
        e, U = np.linalg.eigh(M) # calculate eigenvalues an deigenvectors
        tmp = np.dot(X.T, U).T
        V = tmp[::-1] # reverse since the last eigenvectors are the ones we want
        S = np.sqrt(e)[::-1] #reverse since the last eigenvalues are in increasing order
        for i in range(V.shape[1]):
            V[:,i] /= S
    else:
        # normal PCA, SVD method
        U,S,V = np.linalg.svd(X)
        V = V[:num_data] # only makes sense to return the first num_data
    return V, S, mean_X



path = 'Z:/inhouse/nili/ML/archive/500_train/'
size = 100,100

cat_train = []
for file in os.listdir(os.path.join(path,'cats')):
    if file.endswith("jpg"):
        img = cv2.imread(os.path.join(path,'cats',file),0)
        img = img/255.0
        # converting the type of pixel to float 32
        img = img.astype('float32')
        im = cv2.resize(img,size)
        img=im.flatten()
        cat_train.append(img)
    else:
        continue

cat_train=np.array(cat_train)
V,S,mean=PCA(cat_train)

n_cats=[1, 10, 14, 29, 38, 53, 57, 64, 86, 94, 102, 103, 109, 186,
        194, 199, 202, 210, 213, 219,243, 249, 255, 260, 271, 275,
        301, 308, 320, 332, 336, 349, 356, 384, 385, 403, 406, 418,
        431, 434, 437, 440, 444, 445, 466, 467, 471, 483, 488, 493, 500]

for i in range(len(n_cats)):
    file='cat.'+str(n_cats[i])+'.jpg'
    img = cv2.imread(os.path.join(path,'cats',file),0)
    img = img/255.0
    img = img.astype('float32')
    im = cv2.resize(img,size)
    cat_train.append(im.flatten())
cat_train=np.array(cat_train)
V,S,mean=PCA(cat_train)



fig, axs = plt.subplots(2,4)

plt.subplot(2,4,1)
plt.imshow(mean.reshape(size),cmap='Greys')
plt.title('mean')
plt.axis('off')

for i in range(7):
    plt.subplot(2,4,i+2)
    #axs[i+1].imshow(V[i].reshape(size),cmap='Greys')
    plt.title('mean')
    plt.axis('off')

plt.show()


n_dogs=[2, 4, 8, 10, 13, 30, 33, 51, 56, 67, 68, 76, 77, 79, 82, 
        85, 86, 91, 99, 101, 103,
        104, 114, 117, 123, 125, 131, 133, 149, 152, 155, 156, 166,
        184, 187, 203, 209, 232, 241, 242, 245, 247, 259, 261, 270, 
        274, 275, 278, 309, 310, 311, 328, 339, 353, 355, 358, 364,
        367, 370, 379, 380, 384, 394, 401, 424, 428, 444, 448, 449, 459]
dog_train = []

for i in range(len(n_dogs)):
    file='dog.'+str(n_dogs[i])+'.jpg'
    img = cv2.imread(os.path.join(path,'dogs',file),0)
    img = img/255.0
    img = img.astype('float32')
    im = cv2.resize(img,size)
    dog_train.append(im.flatten())
dog_train=np.array(dog_train)
V,S,mean=PCA(dog_train)



fig, axs = plt.subplots(2,4)

plt.subplot(2,4,1)
plt.imshow(mean.reshape(size),cmap='Greys')
#plt.title('mean')
plt.axis('off')

for i in range(7):
    plt.subplot(2,4,i+2)
    plt.imshow(V[i].reshape(size)+mean.reshape(size),cmap='Greys')
    #plt.title('mean')
    plt.axis('off')

plt.show() 

plt.figure()
plt.imshow(((V[0]*S[0]+V[1]*S[1]+V[2]*S[2]+V[3]*S[3])/sum(S[:-1])).reshape(size),cmap='Greys')
plt.show()