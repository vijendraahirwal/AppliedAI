import torch
import torch.nn as nn
import torch.nn.functional as F
import common_vars as GLOBAL_VARS
class ExperimentWithDepthAndFilter(nn.Module):
    def __init__(self, numOfChannels, numOfClasses):
        super(ExperimentWithDepthAndFilter, self).__init__()

        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels=numOfChannels, out_channels=16, kernel_size=13),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),
            nn.ReLU()
            
        )

        self.cnn2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=13),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.ReLU()
            
        )

        # self.cnn3 = nn.Sequential(
        #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=9),
        #     nn.BatchNorm2d(64),
        #     nn.MaxPool2d(2, 2),
        #     nn.ReLU()
        
        # )

        # self.cnn4 = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5),
        #     nn.BatchNorm2d(128),
        #     nn.MaxPool2d(2, 2),
        #     nn.ReLU()
            
        # )

    

        # self.cnn6 = nn.Sequential(
        #     nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, 2)
        # )

        # self.cnn7 = nn.Sequential(
        #     nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, 2)
        # )

        self.fc1 = nn.Sequential(
            nn.Linear(32*3*3, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.fc3 = nn.Sequential(
            nn.Linear(256, numOfClasses),
            nn.LogSoftmax(dim=1)
        )
        
        

    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        # x = self.cnn3(x)
        # x = self.cnn4(x)
        #x = self.cnn5(x)
        # x = self.cnn6(x)
        # x = self.cnn7(x)
        
        
        x = x.view(x.size(0), -1)
        
        

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
