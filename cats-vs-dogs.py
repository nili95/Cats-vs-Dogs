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

train_path = 'Z:/inhouse/nili/ML/archive/100_train/'
test_path = 'Z:/inhouse/nili/ML/archive/100_test/'
size = 100,100
'''
image_names=[]
y_train = []
x_train = []

def  create_data(path):
    for file in os.listdir(os.path.join(path,'cats')):
        if file.endswith("jpg"):
            image_names.append(os.path.join(path,'cats',file))
            y_train.append(1)
            img = cv2.imread(os.path.join(path,'cats',file))
            im = cv2.resize(img,size)
            x_train.append(im)
        else:
            continue
    for file in os.listdir(os.path.join(path,'dogs')):
        if file.endswith("jpg"):
            image_names.append(os.path.join(path,'dogs',file))
            y_train.append(-1)
            img = cv2.imread(os.path.join(path,'dogs',file))
            im = cv2.resize(img,size)
            x_train.append(im)
        else:
            continue
            
x_train, y_train = shuffle(x_train, y_train)
'''
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
'''

train_transform = transforms.Compose([transforms.Resize((224,224)),normalize])#,torchvision.transforms.Grayscale(1)])


'''
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

test_imgs = ["/cats/cat.4001.jpg",
                        "/cats/cat.4003.jpg",
                        "/dogs/dog.4004.jpg",
                       "/dogs/dog.4006.jpg"]
img_list = [Image.open(test_dir + img_path) for img_path in test_imgs]

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


