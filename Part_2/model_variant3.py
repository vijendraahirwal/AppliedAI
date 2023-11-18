import torch
import torch.nn as nn
import torch.nn.functional as F
import common_vars as GLOBAL_VARS
class ModelVariant3(nn.Module):
    def __init__(self, numOfChannels, numOfClasses):
        super(ModelVariant3, self).__init__()
        

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=numOfChannels, out_channels=10, kernel_size=3),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3),
            nn.BatchNorm2d(10),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=810, out_features=50),
            nn.ReLU()
        )

        self.fc2 = nn.Linear(in_features=50, out_features=numOfClasses)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)

        out = F.dropout(out)
        out = out.view(out.size(0), -1)
        
        out = self.fc1(out)
        out = self.fc2(out)

        return out
