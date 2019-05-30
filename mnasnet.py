from torch import nn


class ConvBNReLU(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        layers = [
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ConvBNReLU, self).__init__(*layers)


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


class MobileInverted(nn.Module):

    def __init__(self, in_channels, out_channels, stride, expand_ratio, kernel_size=3, se_ratio=0, no_skip=False):
        super(MobileInverted, self).__init__()
        self.use_residual = in_channels == out_channels and stride == 1 and not no_skip
        assert stride in [1, 2]
        assert kernel_size in [3, 5]

        hidden_dim = int(in_channels * expand_ratio)
        layers = []

        # pw
        if in_channels != hidden_dim:
            layers += [ConvBNReLU(in_channels, hidden_dim, kernel_size=1)]

        # dw
        layers += [ConvBNReLU(hidden_dim, hidden_dim, kernel_size, stride=stride, groups=hidden_dim)]

        # se
        if se_ratio != 0:
            layers += [SqueezeExcitation(hidden_dim, ratio=se_ratio)]

        # pw-linear
        layers += [
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ]

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MnasNetA1(nn.Module):

    def __init__(self, width_mult=1.0, num_classes=1000):
        super(MnasNetA1, self).__init__()

        settings = [
            # t, c, n, s, k, r
            [1, 16, 1, 1, 3, 0],  # SepConv_3x3
            [6, 24, 2, 2, 3, 0],  # MBConv6_3x3
            [3, 40, 3, 2, 5, 0.25],  # MBConv3_5x5, SE
            [6, 80, 4, 2, 3, 0],  # MBConv6_3x3
            [6, 112, 2, 1, 3, 0.25],  # MBConv6_3x3, SE
            [6, 160, 3, 2, 5, 0.25],  # MBConv6_5x5, SE
            [6, 320, 1, 1, 3, 0]  # MBConv6_3x3
        ]

        features = [ConvBNReLU(3, int(32 * width_mult), 3)]

        in_channels = int(32 * width_mult)
        for i, (t, c, n, s, k, r) in enumerate(settings):
            out_channels = int(c * width_mult)
            no_skip = True if i == 0 else False
            for j in range(n):
                stride = s if j == 0 else 1
                features += [
                    MobileInverted(in_channels, out_channels, stride, t, kernel_size=k, se_ratio=r, no_skip=no_skip)
                ]
                in_channels = out_channels

        features += [ConvBNReLU(in_channels, 1280, kernel_size=1)]
        self.features = nn.Sequential(*features)

        self.avg_pool = nn.AvgPool2d(kernel_size=7)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x
