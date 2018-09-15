## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

import numpy as np


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 4x4 square convolution kernel
        # input size = 224, 224, 1
        # output size = (W-F)/S +1 = 224 - 5 / 1 + 1 = 220
        # after max pool(2,2) = (32, 110, 110)
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv1_bn = nn.BatchNorm2d(32)

        # input size = 110, 110, 1
        # output size = (W-F)/S +1 = 110 - 4 / 1 + 1 = 107
        # after max pool(2,2) = (64, 53, 53)
        self.conv2 = nn.Conv2d(32, 64, 4)
        self.conv2_bn = nn.BatchNorm2d(64)
        # input size = 53, 53, 1
        # output size = (W-F)/S +1 = 53 - 3 / 1 + 1 = 51
        # after max pool (2,2) = (128, 25, 25)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv3_bn = nn.BatchNorm2d(128)
        # input size = 25, 25, 1
        # output size = (W-F)/S +1 = 25 - 2 / 1 + 1 = 24
        # after max pool (3,3) = (256, 8, 8)
        self.conv4 = nn.Conv2d(128, 256, 2)
        self.conv4_bn = nn.BatchNorm2d(256)

        # input size = 8, 8, 1
        # output size = (W-F)/S +1 = 8 - 1 / 1 + 1 = 8
        # after max pool (2,2) = (256, 4, 4)
        self.conv5 = nn.Conv2d(256, 1024, 1)
        self.conv5_bn = nn.BatchNorm2d(1024)

        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool_2 = nn.MaxPool2d(2, 2)
        self.pool_3 = nn.MaxPool2d(3, 3)

        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting

        self.fc1 = nn.Linear(1024*4*4 , 500)

        self.dropout1 = nn.Dropout(p=0.55)

        #self.fc2 = nn.Linear(1000 , 500)

        #self.dropout2 = nn.Dropout(p=0.5)

        self.output = nn.Linear(500, 136)

        self.dropout3 = nn.Dropout(p=0.55)

        ## initialisation of all of the layers
        I.xavier_uniform_(self.conv1.weight, gain=1)
        I.constant_(self.conv1.bias, 0.1)
        I.xavier_uniform_(self.conv2.weight, gain=1)
        I.constant_(self.conv2.bias, 0.1)
        I.xavier_uniform_(self.conv3.weight, gain=1)
        I.constant_(self.conv3.bias, 0.1)
        I.xavier_uniform_(self.conv4.weight, gain=1)
        I.constant_(self.conv4.bias, 0.1)
        I.constant_(self.conv1_bn.weight, 1)
        I.constant_(self.conv1_bn.bias, 0)
        I.constant_(self.conv2_bn.weight, 1)
        I.constant_(self.conv2_bn.bias, 0)
        I.constant_(self.conv3_bn.weight, 1)
        I.constant_(self.conv3_bn.bias, 0)
        I.constant_(self.conv4_bn.weight, 1)
        I.constant_(self.conv4_bn.bias, 0)
        I.uniform_(self.fc1.weight)
        I.constant_(self.fc1.bias, 0.1)
        #I.uniform_(self.fc2.weight)
        #I.constant_(self.fc2.bias, 0.1)
        I.uniform_(self.output.weight)
        I.constant_(self.output.bias, 0.1)

    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x =  F.relu(self.pool_2(self.conv1_bn(self.conv1(x))))
        x =  F.relu(self.pool_2(self.conv2_bn(self.conv2(x))))
        x =  F.relu(self.pool_2(self.conv3_bn(self.conv3(x))))
        x =  F.relu(self.pool_3(self.conv4_bn(self.conv4(x))))
        x =  F.relu(self.pool_2(self.conv5_bn(self.conv5(x))))

        x = x.view(x.size(0), -1)
        x = F.tanh(self.fc1(self.dropout1(x)))
        #x = F.tanh(self.fc2(self.dropout2(x)))
        # a modified x, having gone through all the layers of your model, should be returned
        x = self.output(self.dropout3(x))

        return x
