
import torch.nn as nn

def conv_block(in_channels, out_channels,kernal_size, pool=False):
    layers = [
              nn.Conv2d(in_channels, out_channels, kernel_size=kernal_size, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)
    ]

    if pool:
        layers.append(nn.MaxPool2d(kernel_size=2))
    
    return nn.Sequential(*layers)


