import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelVariant1(nn.Module):
    
    def __init__(self, numOfChannels, numOfClasses):
        super(ModelVariant1, self).__init__()
        
        self.firstConv = nn.Sequential(
            nn.Conv2d(numOfChannels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.secondConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

        self.firstResidual = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.thirdConv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

        self.fourthConv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

        self.secondResidual = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.finalClassifier = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 512),
            nn.Linear(512, numOfClasses)
        )

        self.mainNetwork = nn.Sequential(
            self.firstConv,
            self.secondConv,
            self.firstResidual,
            self.thirdConv,
            self.fourthConv,
            self.secondResidual,
            self.finalClassifier
        )

    def forward(self, inputTensor):
        output = self.firstConv(inputTensor)
        output = self.secondConv(output)
        output = self.firstResidual(output) + output
        output = self.thirdConv(output)
        output = self.fourthConv(output)
        output = self.secondResidual(output) + output
        output = self.finalClassifier(output)
        return output