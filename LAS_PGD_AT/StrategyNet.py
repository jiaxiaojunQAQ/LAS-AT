'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_Strategy(nn.Module):
    def __init__(self, block, num_blocks,  args):
        self.args = args
        super(ResNet_Strategy, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.Attack_method = nn.Linear(512*block.expansion, len(args.attack_types))  # 所用攻击方式的个数 1
        self.Attack_epsilon = nn.Linear(512*block.expansion, len(args.epsilon_types))  # 攻击强度 11
        self.Attack_iters = nn.Linear(512*block.expansion, len(args.attack_iters_types))  # 迭代次数 11
        self.Attack_step_size = nn.Linear(512*block.expansion, len(args.step_size_types))  # 步长
        self.saved_log_probs = []
        self.rewards = []
        self.R1s = []
        self.R2s = []
        self.R3s = []


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        Attack_method = self.Attack_method(out)
        Attack_epsilon = self.Attack_epsilon(out)
        Attack_iters = self.Attack_iters(out)
        Attack_step_size = self.Attack_step_size(out)

        return Attack_method, Attack_epsilon, Attack_iters, Attack_step_size

import argparse
def get_args():
    parser = argparse.ArgumentParser('LAS_AT')
    # target model
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='../cifar-data', type=str)
    parser.add_argument('--out-dir', default='LAS_AT', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--target_model_lr', default=0.2, type=float, help='learning rate')
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--target_model_lr_scheduler', default='cyclic', type=str, choices=['cyclic', 'multistep'])
    parser.add_argument('--target_model_lr_min', default=0., type=float)
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--model', default='ResNet18', type=str, help='model name')

    parser.add_argument('--path', default='RLAT', type=str, help='model name')

    ## search
    parser.add_argument('--attack_types', type=list, default=['IFGSM'], help='all searched policies')
    parser.add_argument('--epsilon_types', type=list, default=range(1, 11))
    parser.add_argument('--attack_iters_types', type=list, default=range(1, 10))
    parser.add_argument('--step_size_types', type=list, default=range(1, 5))

    ## policy Hyperparameters
    parser.add_argument('--policy_model_lr', type=float, default=0.01)
    parser.add_argument('--policy_model_scheduler', default='cyclic', type=str, choices=['cyclic', 'multistep'])
    parser.add_argument('--policy_model_lr_min', default=0., type=float)
    parser.add_argument('--gamma ', type=float, default=0.99)
    parser.add_argument('--controller_hid_size', type=int, default=100)
    parser.add_argument('--max_iter', type=int, default=2)






    arguments = parser.parse_args()
    return arguments
def ResNet18_Strategy(args):
    return ResNet_Strategy(BasicBlock, [2, 2, 2, 2],args)


def ResNet34_Strategy(args):
    return ResNet_Strategy(BasicBlock, [3, 4, 6, 3],args)


def ResNet50_Strategy(args):
    return ResNet_Strategy(Bottleneck, [3, 4, 6, 3],args)


def ResNet101_Strategy(args):
    return ResNet_Strategy(Bottleneck, [3, 4, 23, 3],args)


# def ResNet152(args):
#     return ResNet_policy(Bottleneck, [3, 8, 36, 3],args)



# args = get_args()
# net = ResNet18_policy(args)
# print(net)


# test()
