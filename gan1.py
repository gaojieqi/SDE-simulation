import numpy as np
from keras.datasets import mnist
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data


BATCH_SIZE=1000
IDEA=100
epoch=10

LR_D=0.01
LR_G=0.01


pad=2
s=3
kernelg1=4
kernelg2=3
kernelg3=2


def output_hight_width(w,f,p,s):
    return int((w-f+2*p)/s+1)



(x_train,y_train),(x_test,y_test)=mnist.load_data()
# x_train=x_train.reshape(60000,784)
x_train= torch.from_numpy(x_train)
x_train=Variable(x_train)


w=output_hight_width(28,kernelg1,pad,s)
w=output_hight_width(w,kernelg2,pad,s)
w=output_hight_width(w,kernelg3,pad,s)

torch_dataset=data.TensorDataset(data_tensor=x_train,target_tensor=x_test)
loader=data.Dataloader(torch_dataset=torch_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=4)


G=nn.Sequential(
    nn.ConvTranspose2d(100,64*8,5,3,1),
    nn.BatchNorm2d(64*8),
    nn.ReLU(),

    nn.ConvTranspose2d(100,64*4,5,3,1),
    nn.BatchNorm2d(64*4),
    nn.ReLU(),

    nn.ConvTranspose2d(100,64*2,5,3,1),
    nn.BatchNorm2d(64*2),
    nn.ReLU(),

    nn.ConvTranspose2d(100,64,5,3,1),
    nn.BatchNorm2d(64),
    nn.ReLU(),

    nn.Conv2d(1, 30, kernelg1, stride=s, padding=pad),
    nn.ReLU(),
    nn.MaxPool2d(pad),

    nn.Conv2d(30, 50, kernelg2, stride=s, padding=pad),
    nn.ReLU(),
    nn.MaxPool2d(pad),

    nn.Conv2d(50, 70, kernelg3, stride=s, padding=pad),
    nn.ReLU(),
    nn.MaxPool2d(pad),



                )

D=nn.Sequential(
    nn.Conv2d(1, 30, kernelg1, stride=s, padding=pad),
    nn.ReLU(),
    nn.MaxPool2d(pad),

    nn.Conv2d(30, 50, kernelg2, stride=s, padding=pad),
    nn.ReLU(),
    nn.MaxPool2d(pad),

    nn.Conv2d(50, 70, kernelg3, stride=s, padding=pad),
    nn.ReLU(),
    nn.MaxPool2d(pad),

    nn.Linear(w, 100),
    nn.ReLU(),

    nn.Linear(100, 25),
    nn.ReLU(),

    nn.Linear(25, 2),
    nn.ReLU(),

    nn.LogSoftmax(),

)

opt_D=torch.optim.Adam(D.parameters(),lr=LR_D)
opt_G=torch.optim.Adam(G.parameters(),lr=LR_G)


for epoch in range():
    for step,(batch_x_train,batch_x_test) in enumerate(loader):

        prob_real=D(batch_x_train)
        prob_fake=D()








