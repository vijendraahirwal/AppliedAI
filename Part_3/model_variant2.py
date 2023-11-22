import torch
import torch.nn as nn
import torch.nn.functional as F
import common_vars as GLOBAL_VARS
class ModelVariant2(nn.Module):
    def __init__(self, numOfChannels=1, numOfClasses=4,activation=nn.ReLU(),dropout=0.3):
        super(ModelVariant2, self).__init__()

        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels=numOfChannels, out_channels=16, kernel_size=3),
            nn.BatchNorm2d(16),
            activation,
            nn.MaxPool2d(2, 2)
        )

        self.cnn2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32),
            activation
            
        )

        self.cnn3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            activation
        
        )

        self.cnn4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.BatchNorm2d(128),
            activation,
            nn.MaxPool2d(2, 2)
        )

        self.cnn5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
            nn.BatchNorm2d(128),
            activation,
            nn.MaxPool2d(2, 2)
        )

        
        self.fc1 = nn.Sequential(
            nn.Linear(128*3*3, 512),
            activation,
            nn.Dropout(dropout)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            activation,
            nn.Dropout(dropout)
        )

        self.fc3 = nn.Sequential(
            nn.Linear(256, numOfClasses),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.cnn5(x)
        
        
        x = x.view(x.size(0), -1)
        
        

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
