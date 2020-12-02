# -*- coding: utf-8 -*-
"""
Binary Classifier for Dogs and Cats images
@author: NILI
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import cv2
from sklearn.utils import shuffle
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchsummary import summary
from PIL import Image
from skimage.io import imread
import torch.nn.functional as F

from sklearn.model_selection import train_test_split

# for evaluating the model
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# PyTorch libraries and modules
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD

train_path = 'Z:/inhouse/nili/ML/archive/500_train/'
size = 100,100

image_names=[]


def  create_data(path):
    y_train = []
    x_train = []
    for file in os.listdir(os.path.join(path,'cats')):
        if file.endswith("jpg"):
            image_names.append(os.path.join(path,'cats',file))
            y_train.append(1)
            img = cv2.imread(os.path.join(path,'cats',file),0)
            #img = imread(image_path, as_gray=True)
            # normalizing the pixel values
            img = img/255.0
            # converting the type of pixel to float 32
            img = img.astype('float32')
            im = cv2.resize(img,size)
            x_train.append(im)
        else:
            continue
    for file in os.listdir(os.path.join(path,'dogs')):
        if file.endswith("jpg"):
            image_names.append(os.path.join(path,'dogs',file))
            y_train.append(0)
            img = cv2.imread(os.path.join(path,'dogs',file),0)
            im = cv2.resize(img,size)
            x_train.append(im)
        else:
            continue
    x_train, y_train = shuffle(x_train, y_train)
    return x_train, y_train

            
x_train, y_train = create_data(train_path)

x_train=np.array(x_train)
y_train=np.array(y_train)

#show some plot of the gray scale images
i = 0
plt.figure(figsize=(10,10))
plt.subplot(221), plt.imshow(x_train[i], cmap='gray')
plt.subplot(222), plt.imshow(x_train[i+25], cmap='gray')
plt.subplot(223), plt.imshow(x_train[i+50], cmap='gray')
plt.subplot(224), plt.imshow(x_train[i+75], cmap='gray')


'''
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([transforms.Resize((224,224)),normalize])#,torchvision.transforms.Grayscale(1)])
'''
# create validation set
train_x, val_x, train_y, val_y = train_test_split(x_train, y_train, test_size = 0.1)
(train_x.shape, train_y.shape), (val_x.shape, val_y.shape)

# converting training images into torch format
train_x = train_x.reshape(900, 1, 100, 100)
train_x  = torch.from_numpy(train_x)

# converting the target into torch format
#train_y = train_y.astype();
train_y = torch.from_numpy(train_y)

# shape of training data
train_x.shape, train_y.shape

# converting validation images into torch format
val_x = val_x.reshape(100, 1, 100, 100)
val_x  = torch.from_numpy(val_x)

# converting the target into torch format
#val_y = val_y.astype(torch.long);
val_y = torch.from_numpy(val_y)

# shape of validation data
val_x.shape, val_y.shape



class Net(Module):   
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.linear_layers = Sequential(
            Linear(4 * 25 * 25, 2)
            #Linear(20, 10)
        )
    

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
    

# defining the model
model = Net()
# defining the optimizer
optimizer = Adam(model.parameters(), lr=0.07)
# defining the loss function
loss = CrossEntropyLoss()
# checking if GPU is available
if torch.cuda.is_available():
    model = model.cuda()
    loss = loss.cuda()

summary(model,(1,100,100))
 
print(model)

def train(epoch):
    model.train()
    tr_loss = 0
    # getting the training set
    x_train, y_train = Variable(train_x), Variable(train_y)
    # getting the validation set
    x_val, y_val = Variable(val_x), Variable(val_y)
    y_train = y_train.type(torch.LongTensor)
    y_val = y_val.type(torch.LongTensor)

    # converting the data into GPU format
    if torch.cuda.is_available():
        x_train = x_train.cuda()
        y_train = y_train.cuda()
        x_val = x_val.cuda()
        y_val = y_val.cuda()
    
    # clearing the Gradients of the model parameters
    optimizer.zero_grad()
    
    # prediction for training and validation set
    output_train = model(x_train)
    output_val = model(x_val)

    # computing the training and validation loss
    loss_train = loss(output_train, y_train)
    loss_val = loss(output_val, y_val)
    train_losses.append(loss_train)
    val_losses.append(loss_val)

    # computing the updated weights of all the model parameters
    loss_train.backward()
    optimizer.step()
    tr_loss = loss_train.item()
    if epoch%2 == 0:
        # printing the validation loss
        print('Epoch : ',epoch+1, '\t', 'loss :', loss_val)
        
# defining the number of epochs
n_epochs = 25
# empty list to store training losses
train_losses = []
# empty list to store validation losses
val_losses = []
# training the model
for epoch in range(n_epochs):
    train(epoch)














train_transform = torchvision.transforms.Compose([
    transforms.Resize((224,224)),
        transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
])
train_dataloader = torch.utils.data.DataLoader(dataset=torchvision.datasets.ImageFolder(train_path,
                                                                                       transform=train_transform),
                                              shuffle=True,batch_size=32,num_workers=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

model = torchvision.models.resnet50(pretrained=True).to(device)
for params in model.parameters():
    params.requires_grads = False

model.fc = nn.Sequential(
    nn.Linear(2048,128),
    nn.ReLU(inplace=True),
    nn.Linear(128,2)
).to(device)

criterian = nn.CrossEntropyLoss()
optimizers = torch.optim.Adam(model.fc.parameters())

model_summary = summary(model,(3,224,224))

n_epochs = 4
for epoch in range(n_epochs):
    model.train()
    for batch_idx,(data,labels) in enumerate(train_dataloader):
        data = data.to(device)
        labels = labels.to(device)
        optimizers.zero_grad()
        output = model(data)
        loss = criterian(output,labels)
        loss.backward()
        optimizers.step()
    print(f'epochs: {epoch} loss: {loss.item()}')
    
torch.save(model.state_dict(),'model1.h5')

test_imgs = ["/cats/cat.200.jpg",
                        "/cats/cat.201.jpg",
                        "/dogs/dog.200.jpg",
                       "/dogs/dog.201.jpg"]
img_list = [Image.open( + img_path) for img_path in test_imgs]

test_batch = torch.stack([test_transform(img).to(device)
                                for img in img_list])

pred_logits_tensor = model(test_batch)
pred_logits_tensor

pred_probs = nn.functional.softmax(pred_logits_tensor, dim=1).cpu().data.numpy()
pred_probs

fig, axs = plt.subplots(1, len(img_list), figsize=(20, 5))
for i, img in enumerate(img_list):
    ax = axs[i]
    ax.axis('off')
    ax.set_title("{:.0f}% Cat, {:.0f}% Dog".format(100*pred_probs[i,0],
                                                            100*pred_probs[i,1]))
    ax.imshow(img)


