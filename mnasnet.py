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


class Identity(nn.Module):

    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, *input):
        return input


class MobileInverted(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio, kernel_size=3, se_ratio=0):
        super(MobileInverted, self).__init__()
        self.use_residual = inp == oup and stride == 1

        SEBlock = Identity if se_ratio == 0 else SqueezeExcitation

        layers = []

        hidden_dim = int(inp * expand_ratio)
        if inp != hidden_dim:
            layers.append(ConvBNReLU(inp, hidden_dim, 1))

        layers += [
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, kernel_size, stride=stride, groups=hidden_dim),
            # se
            SqueezeExcitation(hidden_dim, ratio=se_ratio),
            # pw
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            nn.BatchNorm2d(oup)
        ]

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)
