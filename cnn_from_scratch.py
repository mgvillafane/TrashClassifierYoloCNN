import torch
from torch import nn


def conv_block(in_channels, out_channels):
    block = nn.Sequential(
        nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3,
            stride=1,
            padding=1, 
            bias = False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size= 2, stride=2)
    )
    return block




class TrashCNN(nn.Module):
    def __init__(self, num_classes):

        super().__init__()

        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256)
        )

        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(256, num_classes)


    def forward(self, x):

        x = self.features(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x






