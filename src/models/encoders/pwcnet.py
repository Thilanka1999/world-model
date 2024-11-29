from torch import nn


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        ),
        nn.LeakyReLU(0.1),
    )


class PWCEncoder(nn.Module):
    dims = {
        "l1": 16,
        "l2": 32,
        "l3": 64,
        "l4": 96,
        "l5": 128,
        "l6": 196,
    }

    def __init__(self):
        super(PWCEncoder, self).__init__()
        self.conv1 = conv(3, 16, kernel_size=3, stride=2)
        self.conv2 = conv(16, 16, kernel_size=3, stride=1)
        self.conv3 = conv(16, 32, kernel_size=3, stride=2)
        self.conv4 = conv(32, 32, kernel_size=3, stride=1)
        self.conv5 = conv(32, 64, kernel_size=3, stride=2)
        self.conv6 = conv(64, 64, kernel_size=3, stride=1)
        self.conv7 = conv(64, 96, kernel_size=3, stride=2)
        self.conv8 = conv(96, 96, kernel_size=3, stride=1)
        self.conv9 = conv(96, 128, kernel_size=3, stride=2)
        self.conv10 = conv(128, 128, kernel_size=3, stride=1)
        self.conv11 = conv(128, 196, kernel_size=3, stride=2)
        self.conv12 = conv(196, 196, kernel_size=3, stride=1)
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.constant_(m.weight.data, 0.0)
                if m.bias is not None:
                    m.bias.data.zero_()
        """

    def forward(self, img):
        cnv2 = self.conv2(self.conv1(img))
        cnv4 = self.conv4(self.conv3(cnv2))
        cnv6 = self.conv6(self.conv5(cnv4))
        cnv8 = self.conv8(self.conv7(cnv6))
        cnv10 = self.conv10(self.conv9(cnv8))
        cnv12 = self.conv12(self.conv11(cnv10))
        return {
            "l1": cnv2,
            "l2": cnv4,
            "l3": cnv6,
            "l4": cnv8,
            "l5": cnv10,
            "l6": cnv12,
        }
