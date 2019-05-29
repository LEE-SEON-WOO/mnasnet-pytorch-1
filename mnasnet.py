from torch import nn


class ConvBNReLU(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ConvBNReLU, self).__init__(*layers)


class SepConv(nn.Sequential):

    def __init__(self, in_channels, out_channels):
        super(SepConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )


class SqueezeExcitation(nn.Module):

    def __init__(self, num_features, ratio=0.25):
        super(SqueezeExcitation, self).__init__()
        hidden_dim = int(num_features * ratio)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, hidden_dim, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_features, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)
