import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


#hz confirmed: same as WRN28
def mish(x):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function (https://arxiv.org/abs/1908.08681)"""
    return x * torch.tanh(F.softplus(x))


#hz confirmed: same as WRN28
class PSBatchNorm2d(nn.BatchNorm2d):
    """How Does BN Increase Collapsed Neural Network Filters? (https://arxiv.org/abs/2001.11216)"""

    def __init__(self, num_features, alpha=0.1, eps=1e-05, momentum=0.001, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha

    def forward(self, x):
        return super().forward(x) + self.alpha


#hz confirmed: same as WRN28
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.drop_rate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


#hz confirmed: same as WRN28
class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, drop_rate, activate_before_residual))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNetVar(nn.Module):
    def __init__(self, num_classes, depth=28, widen_factor=2, drop_rate=0.0, pn_strength=0.1):
        super(WideResNetVar, self).__init__()
        
        self.pn_strength = pn_strength
        
        channels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor, 128*widen_factor] #hz confirmed: compared to WRN28, added 128*widen_factor

        assert((depth - 4) % 6 == 0)#hz confirmed: the depth arg is actually not used
        n = (depth - 4) / 6 #hz confirmed: the depth arg is actually not used
        
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        
        
        # hz confirmed: compared to WRN28, the stride for 1st block now becomes 2
#         self.block1 = NetworkBlock(
#             n, channels[0], channels[1], block, 1, drop_rate, activate_before_residual=True)
        
        self.block1 = NetworkBlock(
            n, channels[0], channels[1], block, 2, drop_rate, activate_before_residual=True)
        
        
        # 2nd block
        self.block2 = NetworkBlock(
            n, channels[1], channels[2], block, 2, drop_rate)
        # 3rd block
        self.block3 = NetworkBlock(
            n, channels[2], channels[3], block, 2, drop_rate)
        
        #hz confirmed: compared to WRN28, added 4th block, correspondly modify channels[3] --> channels[4] for the following layers
        self.block4 = NetworkBlock(n, channels[3], channels[4], block, 2, drop_rate)
        
        # global average pooling and classifier
#         self.bn1 = nn.BatchNorm2d(channels[3], momentum=0.001)
        self.bn1 = nn.BatchNorm2d(channels[4], momentum=0.001)

        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
#         self.fc = nn.Linear(channels[3], num_classes)
        self.fc = nn.Linear(channels[4], num_classes)
        
#         self.channels = channels[3]
        self.channels = channels[4]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        
        x_copy = out.clone()
#         prCyan('Inside forward, x_copy shape: {} is {}'.format(x_copy.shape, x_copy))
        
        x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)), dim=0 )
#         prCyan('Inside forward, After transformation, x_copy shape: {} is {}'.format(x_copy.shape, x_copy))
        
        out = (1-self.pn_strength) * out + self.pn_strength * x_copy
        
        #hz confirmed: compared to WRN28, added block4
        out = self.block4(out)

        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.channels)
        return self.fc(out)


def build_wideresnet(depth, widen_factor, dropout, num_classes, pn_strength):
#     logger.info(f"Model: WideResNet {depth}x{widen_factor}")
    logger.info(f"Model: WideResNet-37-2")

    return WideResNetVar(depth=depth,
                      widen_factor=widen_factor,
                      drop_rate=dropout,
                      num_classes=num_classes,
                      pn_strength=pn_strength)