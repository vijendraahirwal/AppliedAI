
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelVariant3(nn.Module):
    def __init__(self, numOfChannels, numOfClasses):
        super(ModelVariant3, self).__init__()

        self.primaryConv = nn.Sequential(
            nn.Conv2d(in_channels=numOfChannels, out_channels=10, kernel_size=3),
            nn.ReLU()
        )

        self.secondaryConv = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.tertiaryConv = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3),
            nn.BatchNorm2d(10),
            nn.ReLU()
        )

        self.quaternaryConv = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.denseLayer1 = nn.Sequential(
            nn.Linear(in_features=810, out_features=50),
            nn.ReLU()
        )

        self.finalDenseLayer = nn.Linear(in_features=50, out_features=numOfClasses)

    def forward(self, inputData):
        layerOutput = self.primaryConv(inputData)
        layerOutput = self.secondaryConv(layerOutput)
        layerOutput = self.tertiaryConv(layerOutput)
        layerOutput = self.quaternaryConv(layerOutput)

        layerOutput = F.dropout(layerOutput)
        layerOutput = layerOutput.view(layerOutput.size(0), -1)
        
        layerOutput = self.denseLayer1(layerOutput)
        layerOutput = self.finalDenseLayer(layerOutput)

        return layerOutput
