import torch
import torch.nn as nn

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride, downsample):
        super(BasicBlock, self).__init__()
        self.bn = nn.Sequential(nn.BatchNorm2d(in_planes), nn.ReLU())
        self.conv = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1),
                                  nn.BatchNorm2d(planes), nn.ReLU(),
                                  nn.Dropout(p=dropout_rate),
                                  nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1))
        self.downsample = downsample

    def forward(self, x):
        bn_x = self.bn(x)
        out = self.conv(bn_x)
        if self.downsample:
            out += self.downsample(bn_x)
        else:
            out += x
        return out


class WRN(nn.Module):
    def __init__(self,
                 depth,
                 widen_factor,
                 dropout_rate,
                 in_planes=3,
                 num_classes=10,
                 dataset='cifar10',
                 num_input_channels=3):
        super(WRN, self).__init__()
        if 'cifar10' in dataset:
            mean=CIFAR10_MEAN
            std=CIFAR10_STD
        elif 'cifar100' in dataset:
            mean=CIFAR100_MEAN
            std=CIFAR100_STD
        else:
            mean=IMAGENET_MEAN
            std=IMAGENET_STD
        self.in_planes = in_planes
        N, k = (depth - 4) // 6, widen_factor

        layers = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = nn.Conv2d(self.in_planes, layers[0], kernel_size=3, stride=1, padding=1)
        self.in_planes = layers[0]
        self.conv2 = self._make_layer(layers[1], N, dropout_rate, stride=1)
        self.conv3 = self._make_layer(layers[2], N, dropout_rate, stride=2)
        self.conv4 = self._make_layer(layers[3], N, dropout_rate, stride=2)
        self.bn = nn.Sequential(nn.BatchNorm2d(layers[3]), nn.ReLU())
        self.pool = nn.AvgPool2d(kernel_size=8)
        self.fc = nn.Linear(layers[3], num_classes)

        self.mean = torch.tensor(mean).view(num_input_channels, 1, 1)
        self.std = torch.tensor(std).view(num_input_channels, 1, 1)
        self.mean_cuda = None
        self.std_cuda = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.uniform_(m.weight, 0, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, num_blocks, dropout_rate, stride):
        layers = []
        downsample = nn.Conv2d(self.in_planes, planes, kernel_size=1, stride=stride)
        layers.append(BasicBlock(self.in_planes, planes, dropout_rate, stride, downsample))

        self.in_planes = planes
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(planes, planes, dropout_rate, stride=1, downsample=None))

        return nn.Sequential(*layers)

    def forward(self, x):
        if x.is_cuda:
            if self.mean_cuda is None:
                self.mean_cuda = self.mean.cuda()
                self.std_cuda = self.std.cuda()
            out = (x - self.mean_cuda) / self.std_cuda
        else:
            out = (x - self.mean) / self.std
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.bn(out)
        out = self.pool(out)
        features = out.view(out.size(0), -1)
        out = self.fc(features)
        return out, features