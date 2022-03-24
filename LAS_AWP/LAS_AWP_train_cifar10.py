import argparse
import logging
import sys
import time
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import *



import os
import torch.optim as optim
from CIFAR10_models import *
from collections import namedtuple
from utils_awp import AdvWeightPerturb
from torch.distributions import Categorical, Bernoulli
import copy
import os
import numpy
import argparse
import logging
from StrategyNet import *

cifar10_mean = (0.4914, 0.4822, 0.4465) # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616) # equals np.std(train_set.train_data, axis=(0,1,2))/255

def normalise(x, mean=cifar10_mean, std=cifar10_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean*255
    x *= 1.0/(255*std)
    return x

def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border), (0, 0)], mode='reflect')

def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target])

mu = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
std = torch.tensor(cifar10_std).view(3, 1, 1).cuda()
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Crop(namedtuple('Crop', ('h', 'w'))):
    def __call__(self, x, x0, y0):
        return x[:, y0:y0 + self.h, x0:x0 + self.w]

    def options(self, x_shape):
        C, H, W = x_shape
        return {'x0': range(W + 1 - self.w), 'y0': range(H + 1 - self.h)}

    def output_shape(self, x_shape):
        C, H, W = x_shape
        return (C, self.h, self.w)


class FlipLR(namedtuple('FlipLR', ())):
    def __call__(self, x, choice):
        return x[:, :, ::-1].copy() if choice else x

    def options(self, x_shape):
        return {'choice': [True, False]}


class Cutout(namedtuple('Cutout', ('h', 'w'))):
    def __call__(self, x, x0, y0):
        x = x.copy()
        x[:, y0:y0 + self.h, x0:x0 + self.w].fill(0.0)
        return x

    def options(self, x_shape):
        C, H, W = x_shape
        return {'x0': range(W + 1 - self.w), 'y0': range(H + 1 - self.h)}


class Transform():
    def __init__(self, dataset, transforms):
        self.dataset, self.transforms = dataset, transforms
        self.choices = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, labels = self.dataset[index]
        for choices, f in zip(self.choices, self.transforms):
            args = {k: v[index] for (k, v) in choices.items()}
            data = f(data, **args)
        return data, labels

    def set_random_choices(self):
        self.choices = []
        x_shape = self.dataset[0][0].shape
        N = len(self)
        for t in self.transforms:
            options = t.options(x_shape)
            x_shape = t.output_shape(x_shape) if hasattr(t, 'output_shape') else x_shape
            self.choices.append({k: np.random.choice(v, size=N) for (k, v) in options.items()})


#####################
## dataset
#####################
import torchvision


def cifar10(root):
    train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=True)
    return {
        'train': {'data': train_set.data, 'labels': train_set.targets},
        'test': {'data': test_set.data, 'labels': test_set.targets}
    }




class Batches():
    def __init__(self, dataset, batch_size, shuffle, set_random_choices=False, num_workers=0, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.set_random_choices = set_random_choices
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=shuffle,
            drop_last=drop_last
        )

    def __iter__(self):
        if self.set_random_choices:
            self.dataset.set_random_choices()
        return ({'input': x.to(device).half(), 'target': y.to(device).long()} for (x, y) in self.dataloader)

    def __len__(self):
        return len(self.dataloader)


def normalize(X):

    return (X - mu)/std


upper_limit, lower_limit = 1,0


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


class Batches():
    def __init__(self, dataset, batch_size, shuffle, set_random_choices=False, num_workers=0, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.set_random_choices = set_random_choices
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=shuffle, drop_last=drop_last
        )

    def __iter__(self):
        if self.set_random_choices:
            self.dataset.set_random_choices()
        return ({'input': x.cuda().float(), 'target': y.cuda().long()} for (x,y) in self.dataloader)

    def __len__(self):
        return len(self.dataloader)


def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts,
               norm= "l_inf", early_stop=False,
               mixup=False, y_a=None, y_b=None, lam=None):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*epsilon
        else:
            raise ValueError
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(normalize(X + delta))
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            if mixup:
                criterion = nn.CrossEntropyLoss()
                loss = mixup_criterion(criterion, model(normalize(X+delta)), y_a, y_b, lam)
            else:
                loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
        if mixup:
            criterion = nn.CrossEntropyLoss(reduction='none')
            all_loss = mixup_criterion(criterion, model(normalize(X+delta)), y_a, y_b, lam)
        else:
            all_loss = F.cross_entropy(model(normalize(X+delta)), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='WideResNet')
    parser.add_argument('--l2', default=0, type=float)
    parser.add_argument('--l1', default=0, type=float)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--batch-size-test', default=128, type=int)
    parser.add_argument('--data-dir', default='../cifar-data', type=str)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr-schedule', default='piecewise', choices=['superconverge', 'piecewise', 'linear', 'piecewisesmoothed', 'piecewisezoom', 'onedrop', 'multipledecay', 'cosine', 'cyclic'])
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--lr-one-drop', default=0.01, type=float)
    parser.add_argument('--lr-drop-epoch', default=100, type=int)
    parser.add_argument('--attack', default='pgd', type=str, choices=['pgd', 'fgsm', 'free', 'none'])
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--attack-iters', default=10, type=int)
    parser.add_argument('--attack-iters-test', default=20, type=int)
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--pgd-alpha', default=2, type=float)
    parser.add_argument('--fgsm-alpha', default=1.25, type=float)
    parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'])
    parser.add_argument('--fgsm-init', default='random', choices=['zero', 'random', 'previous'])
    parser.add_argument('--out-dir', default='cifar_model', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--width-factor', default=10, type=int)
    parser.add_argument('--resume', default=0, type=int)
    parser.add_argument('--cutout', action='store_true')
    parser.add_argument('--cutout-len', type=int)
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--mixup-alpha', type=float)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--chkpt-iters', default=10, type=int)
    parser.add_argument('--awp-gamma', default=0.01, type=float)
    parser.add_argument('--awp-warmup', default=0, type=int)

    ## search
    parser.add_argument('--attack_types', type=list, default=['IFGSM'], help='all searched policies')
    parser.add_argument('--epsilon_types', type=int, nargs="*", default=range(7, 15))
    parser.add_argument('--attack_iters_types', type=int, nargs="*", default=range(8, 16))
    parser.add_argument('--step_size_types', type=int, nargs="*", default=range(2, 5))


    parser.add_argument('--policy_model_lr', type=float, default=0.0001)
    parser.add_argument('--policy_model_lr_scheduler', default='multistep', type=str, choices=['cyclic', 'multistep'])
    parser.add_argument('--policy_model_lr_min', default=0., type=float)
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--factor', default=0.7, type=float, help='Label Smoothing')
    parser.add_argument('--a', default=1, type=float)
    parser.add_argument('--b', default=1, type=float)
    parser.add_argument('--c', default=1, type=float)

    parser.add_argument('--R2_param', default=1, type=float)
    parser.add_argument('--R3_param', default=1, type=float)
    parser.add_argument('--clip_grad_norm', default=1.0, type=float)

    return parser.parse_args()

eps = np.finfo(np.float32).eps.item()
start_epsilon = (8.0 / 255.)
start_alpha = (2.0 / 255.)
def _label_smoothing(label, factor):
    one_hot = np.eye(10)[label.cuda().data.cpu().numpy()]

    result = one_hot * factor + (one_hot - 1.) * ((factor - 1) / float(10 - 1))

    return result


def LabelSmoothLoss(input, target):
    log_prob = F.log_softmax(input, dim=-1)
    loss = (-target * log_prob).sum(dim=-1).mean()
    return loss
def _get_sub_policies(attack_id_list, espilon_id_list, attack_iters_id_list, step_size_id_list, prob_id_list, args):
    policies = []
    attack_id_list = attack_id_list[0].cpu().numpy()
    espilon_id_list = espilon_id_list[0].cpu().numpy()
    attack_iters_id_list = attack_iters_id_list[0].cpu().numpy()
    step_size_id_list = step_size_id_list[0].cpu().numpy()

    for n in range(args.subpolicy_num):
        sub_policy = {}
        for i in range(args.op_num_pre_subpolicy):
            all_policy = {}
            # print(n+i)
            # print(args.epsilon_types)
            # print(espilon_id_list[n+i])
            all_policy['attack'] = args.attack_types[attack_id_list[n + i]]
            all_policy['epsilon'] = args.epsilon_types[espilon_id_list[n + i]]
            all_policy['attack_iters'] = args.attack_iters_types[attack_iters_id_list[n + i]]
            all_policy['step_size'] = args.step_size_types[step_size_id_list[n + i]]

            sub_policy[i] = all_policy
        policies.append(sub_policy)
    return policies


def _get_all_policies(attack_id_list, espilon_id_list, attack_iters_id_list, step_size_id_list, args):
    policies = []
    #print(attack_id_list)
    attack_id_list = attack_id_list[0].cpu().numpy()
    espilon_id_list = espilon_id_list[0].cpu().numpy()
    attack_iters_id_list = attack_iters_id_list[0].cpu().numpy()
    step_size_id_list = step_size_id_list[0].cpu().numpy()
    # prob_id_list=prob_id_list[0].cpu().numpy()
    for n in range(len(attack_id_list)):
        sub_policy = {}

        all_policy = {}
        # print(n+i)
        # print(args.epsilon_types)
        # print(espilon_id_list[n+i])
        all_policy['attack'] = args.attack_types[attack_id_list[n]]
        all_policy['epsilon'] = args.epsilon_types[espilon_id_list[n]]

        all_policy['attack_iters'] = args.attack_iters_types[attack_iters_id_list[n]]

        all_policy['step_size'] = args.step_size_types[step_size_id_list[n]]
        # all_policy['prob'] = args.prob_types[prob_id_list[n]]
        sub_policy[n] = all_policy
        policies.append(sub_policy)

    return policies
def select_action(policy_model, state,args):
    # policy_model = policy_model.eval()

    outputs = policy_model(state)
    attack_id_list = []
    espilon_id_list = []
    attack_iters_id_list = []
    step_size_id_list = []
    prob_list = []
    action_list = []

    max_attack_id_list = []
    max_espilon_id_list = []
    max_attack_iters_id_list = []
    max_step_size_id_list = []
    # max_prob_list = []
    # max_action_list = []
    temp_saved_log_probs = []
    for id in range(4):

        logits = outputs[id]
        probs = F.softmax(logits, dim=-1)
        max_probs = probs.data.clone()
        m = Categorical(probs)

        prob_list.append(m)
        action = m.sample()

        max_action = max_probs.max(1)[1]
        # print(action.shape)
        mode = id % 5
        if mode == 0:
            attack_id_list.append(action)
            max_attack_id_list.append(max_action)
        elif mode == 1:
            espilon_id_list.append(action)
            max_espilon_id_list.append(max_action)
        elif mode == 2:
            attack_iters_id_list.append(action)
            max_attack_iters_id_list.append(max_action)
        elif mode == 3:
            step_size_id_list.append(action)
            max_step_size_id_list.append(max_action)
        temp_saved_log_probs.append(m.log_prob(action))
    policy_model.saved_log_probs.append(temp_saved_log_probs)
    curpolicy = _get_all_policies(attack_id_list, espilon_id_list, attack_iters_id_list, step_size_id_list, args)
    max_curpolicy = _get_all_policies(max_attack_id_list, max_espilon_id_list, max_attack_iters_id_list,
                                      max_step_size_id_list, args)
    action_list.append(attack_id_list)
    action_list.append(espilon_id_list)
    action_list.append(attack_iters_id_list)
    action_list.append(step_size_id_list)


    return action_list, curpolicy, prob_list, max_curpolicy







def Attack_policy_batch(input_batch, y_batch, target_model, policies):
    criterion = nn.CrossEntropyLoss()
    X, y = input_batch.cuda(), y_batch.cuda()
    delta_batch = torch.zeros_like(X).cuda()
    std = torch.tensor( (1.0, 1.0, 1.0)).view(3, 1, 1).cuda()
    alpha_batch = []
    epsilon_batch = []
    attack_iters_batch = []
    for ii in range(len(policies)):
        epsilon = (policies[ii][ii]['epsilon'] / 255.)/std
        epsilon_batch.append(epsilon.cpu().numpy())

        alpha = (policies[ii][ii]['step_size'] / 255.)/std
        alpha_batch.append(alpha.cpu().numpy())
        attack_iters = policies[ii][ii]['attack_iters']
        temp_batch = torch.randint(attack_iters, attack_iters + 1, (3, 1, 1))
        attack_iters_batch.append(temp_batch.cpu().numpy())
    alpha_batch = torch.from_numpy(numpy.array(alpha_batch)).cuda()
    epsilon_batch = torch.from_numpy(numpy.array(epsilon_batch)).cuda()
    attack_iters_batch = torch.from_numpy(numpy.array(attack_iters_batch)).cuda()

    max_attack_iters = torch.max(attack_iters_batch).cpu().numpy()
    # print(torch.max(attack_iters_batch))
    delta_batch.uniform_(-start_epsilon, start_epsilon)
    delta_batch = clamp(delta_batch, lower_limit - X, upper_limit - X)
    delta_batch.requires_grad = True
    for _ in range(max_attack_iters):
        output = target_model(normalize(X + delta_batch))


        mask_bacth = attack_iters_batch.ge(1).float()
        # print(alpha_batch[0])

        loss = criterion(output, y)

        loss.backward()

        grad = delta_batch.grad.detach()


        delta_batch.data = clamp(delta_batch + mask_bacth * alpha_batch * torch.sign(grad), -epsilon_batch,
                                 epsilon_batch)
        #delta_batch.data = clamp(delta_batch, lower_limit - X, upper_limit - X)
        delta_batch.data = clamp(delta_batch, lower_limit - X, upper_limit - X)
        attack_iters_batch = attack_iters_batch - 1
        delta_batch.grad.zero_()
    # print( lower_limit.shape)
    # print( torch.sign(grad).shape)
    delta_batch = delta_batch.detach()

    return delta_batch








def train_target_model(input_batch, y_batch, copy_target_model,proxy,args,epoch,lr):

    proxy_opt = torch.optim.SGD(proxy.parameters(), lr=0.01)
    awp_adversary = AdvWeightPerturb(model=copy_target_model, proxy=proxy, proxy_optim=proxy_opt, gamma=args.awp_gamma)
    if epoch >= args.awp_warmup:
        # not compatible to mixup currently.
        assert (not args.mixup)
        awp = awp_adversary.calc_awp(inputs_adv=input_batch,
                                     targets=y_batch)
        awp_adversary.perturb(awp)


    X, Y = input_batch.cuda(), y_batch.cuda()
    label_smoothing = Variable(torch.tensor(_label_smoothing(Y, args.factor)).cuda())
    target_lr =lr
    optimizer = optim.SGD(copy_target_model.parameters(), lr=target_lr, momentum=0.9, weight_decay=5e-4)
    copy_target_model.train()
    optimizer.zero_grad()
    target_output = copy_target_model(normalize(X))
    copy_target_loss = LabelSmoothLoss(target_output, label_smoothing.float())
    copy_target_loss.backward()
    optimizer.step()
    if epoch >= args.awp_warmup:
        awp_adversary.restore(awp)
    return copy_target_model
def Get_delta(input_batch, y_batch, target_model, action):
    target_model.eval()
    inputs, targets = input_batch.cuda(), y_batch.cuda()
    delta = Attack_policy_batch(input_batch, y_batch, target_model, action)
    return inputs + delta
def Get_reward(input_batch, y_batch, target_model, action,proxy,args,epoch, lr):
    target_model.eval()
    criterion = nn.CrossEntropyLoss()
    inputs, targets = input_batch.cuda(), y_batch.cuda()
    delta = Attack_policy_batch(input_batch, y_batch, target_model, action)
    with torch.no_grad():
        ori_clean_output = target_model(inputs)
        output = target_model(inputs + delta)
    # logsoftmax_func = nn.LogSoftmax(dim=1)
    # soft_output = logsoftmax_func(output)
    # y_one_hot = F.one_hot(y_batch, 10).float()
    # print(y_one_hot.shape)
    R1 = criterion(output, targets)  #### R1的奖励函数
    R1 = torch.clamp(R1, 0, 10)

    #ori_R3 = criterion(ori_clean_output, targets)  #### R1的奖励函数
    #ori_R3 = (ori_clean_output.max(1)[1] == targets).sum().item()
    # epsilon = (8 / 255.)
    # alpha = (2 / 255.)
    # ori_pgd_delta = attack_pgd(target_model, inputs, targets, epsilon, alpha, 10, 2)
    # target_model.eval()
    # with torch.no_grad():
    #     ori_R2_output=target_model(inputs + ori_pgd_delta)

    #ori_R2=criterion(ori_R2_output, targets)
    #ori_R2 = (ori_R2_output.max(1)[1] == targets).sum().item()


    copy_target_model = copy.deepcopy(target_model)
    copy_target_model.train()
    #train_target_model(input_batch, y_batch, copy_target_model, proxy, args, epoch, lr, proxy_lr)

    copy_target_model = train_target_model(inputs + delta, targets, copy_target_model,proxy,args,epoch, lr)
    epsilon = (8 / 255.)
    alpha = (2 / 255.)
    pgd_delta = attack_pgd(copy_target_model, inputs, targets, epsilon, alpha, 10, 2)
    copy_target_model.eval()
    with torch.no_grad():
        R2_output = copy_target_model(inputs + pgd_delta)
        clean_output = copy_target_model(inputs)
    # # logsoftmax_func = nn.LogSoftmax(dim=1)
    # # soft_output = logsoftmax_func(output)
    # # y_one_hot = F.one_hot(y_batch, 10).float()
    # # print(y_one_hot.shape)
    # R2 = criterion(R2_output, targets) #### R2的奖励函数
    # R3=criterion(clean_output, targets)
    R2 = (R2_output.max(1)[1] == targets).sum().item()
    R3 = (clean_output.max(1)[1] == targets).sum().item()

    test_n = targets.size(0)
    R2=(R2)/test_n*args.R2_param
    R3=(R3)/test_n*args.R3_param

    R2 = torch.clamp(torch.tensor(R2), -10, 10)
    R3 = torch.clamp(torch.tensor(R3), -10, 10)
    print('R1:', R1)

    print("R2:", R2)

    print("R3:", R3)
    return (args.a*R1 + args.b*R2+args.c*R3), R1, R2, R3,inputs + delta


def main():
    args = get_args()
    if args.awp_gamma <= 0.0:
        args.awp_warmup = np.infty

    out_dir = os.path.join(args.out_dir, 'model_' + args.model)


    out_dir=os.path.join(out_dir,'epsilon_types_'+str(min(args.epsilon_types))+'_'+str(max(args.epsilon_types)))
    out_dir=os.path.join(out_dir,'attack_iters_types_'+str(min(args.attack_iters_types))+'_'+str(max(args.attack_iters_types)))
    out_dir=os.path.join(out_dir,'step_size_types_'+str(min(args.step_size_types))+'_'+str(max(args.step_size_types)))
    #
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(out_dir, 'eval.log' if args.eval else 'output.log')),
            logging.StreamHandler()
        ])

    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    transforms = [Crop(32, 32), FlipLR()]
    if args.cutout:
        transforms.append(Cutout(args.cutout_len, args.cutout_len))
    if args.val:
        try:
            dataset = torch.load("cifar10_validation_split.pth")
        except:
            print("Couldn't find a dataset with a validation split, did you run "
                  "generate_validation.py?")
            return
        val_set = list(zip(transpose(dataset['val']['data']/255.), dataset['val']['labels']))
        val_batches = Batches(val_set, args.batch_size, shuffle=False, num_workers=2)
    else:
        dataset = cifar10(args.data_dir)
    train_set = list(zip(transpose(pad(dataset['train']['data'], 4)/255.),
        dataset['train']['labels']))
    train_set_x = Transform(train_set, transforms)
    train_batches = Batches(train_set_x, args.batch_size, shuffle=True, set_random_choices=True, num_workers=2)

    test_set = list(zip(transpose(dataset['test']['data']/255.), dataset['test']['labels']))
    test_batches = Batches(test_set, args.batch_size_test, shuffle=False, num_workers=2)

    epsilon = (args.epsilon / 255.)
    pgd_alpha = (args.pgd_alpha / 255.)

    if args.model == 'PreActResNet18':
        model = PreActResNet18()
        proxy = PreActResNet18()
    elif args.model == "ResNet18":
        model = ResNet18()
        proxy = ResNet18()
    elif args.model == 'WideResNet':
        model = WideResNet()
        proxy = WideResNet()
    else:
        raise ValueError("Unknown model")

    model = nn.DataParallel(model).cuda()
    proxy = nn.DataParallel(proxy).cuda()

    Strategy_model = ResNet18_Strategy(args)

    Strategy_model.cuda()
    Strategy_model.train()
    print(Strategy_model)
    # policy_optimizer = optim.SGD(Policy_model.parameters(), lr=args.policy_model_lr)
    policy_optimizer = optim.SGD([{'params': Strategy_model.parameters(), 'initial_lr': args.policy_model_lr}],
                                 lr=args.policy_model_lr)

    if args.l2:
        decay, no_decay = [], []
        for name,param in model.named_parameters():
            if 'bn' not in name and 'bias' not in name:
                decay.append(param)
            else:
                no_decay.append(param)
        params = [{'params':decay, 'weight_decay':args.l2},
                  {'params':no_decay, 'weight_decay': 0 }]
    else:
        params = model.parameters()

    opt = torch.optim.SGD(params, lr=args.lr_max, momentum=0.9, weight_decay=5e-4)

    proxy_opt = torch.optim.SGD(proxy.parameters(), lr=0.01)
    awp_adversary = AdvWeightPerturb(model=model, proxy=proxy, proxy_optim=proxy_opt, gamma=args.awp_gamma)

    criterion = nn.CrossEntropyLoss()

    if args.attack == 'free':
        delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()
        delta.requires_grad = True
    elif args.attack == 'fgsm' and args.fgsm_init == 'previous':
        delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()
        delta.requires_grad = True

    if args.attack == 'free':
        epochs = int(math.ceil(args.epochs / args.attack_iters))
    else:
        epochs = args.epochs


    if args.lr_schedule == 'superconverge':
        lr_schedule = lambda t: np.interp([t], [0, args.epochs * 2 // 5, args.epochs], [0, args.lr_max, 0])[0]
    elif args.lr_schedule == 'piecewise':
        def lr_schedule(t):
            if t / args.epochs < 0.5:
                return args.lr_max, args.policy_model_lr
            elif t / args.epochs < 0.75:
                return args.lr_max / 10., args.policy_model_lr/10.
            else:
                return args.lr_max / 100.,args.policy_model_lr/100.


    elif args.lr_schedule == 'linear':
        lr_schedule = lambda t: np.interp([t], [0, args.epochs // 3, args.epochs * 2 // 3, args.epochs], [args.lr_max, args.lr_max, args.lr_max / 10, args.lr_max / 100])[0]
    elif args.lr_schedule == 'onedrop':
        def lr_schedule(t):
            if t < args.lr_drop_epoch:
                return args.lr_max
            else:
                return args.lr_one_drop
    elif args.lr_schedule == 'multipledecay':
        def lr_schedule(t):
            return args.lr_max - (t//(args.epochs//10))*(args.lr_max/10)
    elif args.lr_schedule == 'cosine':
        def lr_schedule(t):
            return args.lr_max * 0.5 * (1 + np.cos(t / args.epochs * np.pi))
    elif args.lr_schedule == 'cyclic':
        lr_schedule = lambda t: np.interp([t], [0, 0.4 * args.epochs, args.epochs], [0, args.lr_max, 0])[0]







    best_test_robust_acc = 0
    best_val_robust_acc = 0
    if args.resume:
        start_epoch = args.resume
        model.load_state_dict(torch.load(os.path.join(out_dir, f'model_{start_epoch-1}.pth')))
        opt.load_state_dict(torch.load(os.path.join(out_dir, f'opt_{start_epoch-1}.pth')))
        logger.info(f'Resuming at epoch {start_epoch}')

        if os.path.exists(os.path.join(out_dir, f'model_best.pth')):
            best_test_robust_acc = torch.load(os.path.join(out_dir, f'model_best.pth'))['test_robust_acc']
        if args.val:
            best_val_robust_acc = torch.load(os.path.join(out_dir, f'model_val.pth'))['val_robust_acc']
    else:
        start_epoch = 0

    if args.eval:
        if not args.resume:
            logger.info("No model loaded to evaluate, specify with --resume FNAME")
            return
        logger.info("[Evaluation mode]")

    logger.info('Epoch \t Train Time \t Test Time \t LR \t \t Train Loss \t Train Acc \t Train Robust Loss \t Train Robust Acc \t Test Loss \t Test Acc \t Test Robust Loss \t Test Robust Acc')
    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        train_loss = 0
        train_acc = 0
        train_robust_loss = 0
        train_robust_acc = 0
        train_n = 0
        for i, batch in enumerate(train_batches):
            if args.eval:
                break
            X, y = batch['input'], batch['target']
            if args.mixup:
                X, y_a, y_b, lam = mixup_data(X, y, args.mixup_alpha)
                X, y_a, y_b = map(Variable, (X, y_a, y_b))
            lr,policy_lr = lr_schedule(epoch + (i + 1) / len(train_batches))
            opt.param_groups[0].update(lr=lr)
            policy_optimizer.param_groups[0].update(lr=policy_lr)
            pocliy_inputs = X.clone().cuda()
            if args.attack == 'pgd':
                # Random initialization
                if args.mixup:
                    delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm, mixup=True, y_a=y_a, y_b=y_b, lam=lam)
                else:
                    Strategy_model.train()


                    action_list, policy_outputs, policy_prob, max_policy_outputs = select_action(Strategy_model,
                                                                                                 pocliy_inputs,args)
                    # Get_reward(input_batch, y_batch, target_model, action,proxy,args,epoch, lr, proxy_lr)
                    reward, R1, R2, R3, adv_examples = Get_reward(pocliy_inputs, y, model, policy_outputs,proxy,args,epoch, lr)
                    log_probs = []
                    policy_loss = []


                    for j in range(4):
                        log_probs.append(policy_prob[j].log_prob(action_list[j][0]))
                        policy_loss.append(-log_probs[j] * reward)
                        # print(action_list)
                        # logger.info(action_list)
                    policy_loss = (policy_loss[0].mean() + policy_loss[1].mean() + policy_loss[2].mean() + policy_loss[
                        3].mean())

                    policy_optimizer.zero_grad()

                    policy_loss.backward()
                    torch.nn.utils.clip_grad_norm_(Strategy_model.parameters(), args.clip_grad_norm)
                    policy_optimizer.step()

                    pocliy_inputs1 = X.clone().cuda()
                    with torch.no_grad():
                        Strategy_model.eval()

                        action_list, policy_outputs, policy_prob, max_policy_outputs = select_action(Strategy_model,
                                                                                                         pocliy_inputs1,args)
                        print(policy_outputs)

                            # print(policy_outputs)a

                    adv_examples = Get_delta(pocliy_inputs, y, model, policy_outputs)
                    pocliy_inputs1 = adv_examples
                    delta = torch.clamp(pocliy_inputs1-X, min=-epsilon, max=epsilon)











                    #delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm)
                delta = delta.detach()
            elif args.attack == 'fgsm':
                delta = attack_pgd(model, X, y, epsilon, args.fgsm_alpha*epsilon, 1, 1, args.norm)
            # Standard training
            elif args.attack == 'none':
                delta = torch.zeros_like(X)
            X_adv = normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit))

            model.train()
            # calculate adversarial weight perturbation and perturb it
            if epoch >= args.awp_warmup:
                # not compatible to mixup currently.
                assert (not args.mixup)
                awp = awp_adversary.calc_awp(inputs_adv=X_adv,
                                             targets=y)
                awp_adversary.perturb(awp)

            robust_output = model(X_adv)
            if args.mixup:
                robust_loss = mixup_criterion(criterion, robust_output, y_a, y_b, lam)
            else:
                robust_loss = criterion(robust_output, y)

            if args.l1:
                for name,param in model.named_parameters():
                    if 'bn' not in name and 'bias' not in name:
                        robust_loss += args.l1*param.abs().sum()

            opt.zero_grad()
            robust_loss.backward()
            opt.step()

            if epoch >= args.awp_warmup:
                awp_adversary.restore(awp)

            output = model(normalize(X))
            if args.mixup:
                loss = mixup_criterion(criterion, output, y_a, y_b, lam)
            else:
                loss = criterion(output, y)

            train_robust_loss += robust_loss.item() * y.size(0)
            train_robust_acc += (robust_output.max(1)[1] == y).sum().item()
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)

        train_time = time.time()

        model.eval()
        test_loss = 0
        test_acc = 0
        test_robust_loss = 0
        test_robust_acc = 0
        test_n = 0
        for i, batch in enumerate(test_batches):
            X, y = batch['input'], batch['target']

            # Random initialization
            if args.attack == 'none':
                delta = torch.zeros_like(X)
            else:
                delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters_test, args.restarts, args.norm, early_stop=args.eval)
            delta = delta.detach()

            robust_output = model(normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
            robust_loss = criterion(robust_output, y)

            output = model(normalize(X))
            loss = criterion(output, y)

            test_robust_loss += robust_loss.item() * y.size(0)
            test_robust_acc += (robust_output.max(1)[1] == y).sum().item()
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            test_n += y.size(0)

        test_time = time.time()

        if args.val:
            val_loss = 0
            val_acc = 0
            val_robust_loss = 0
            val_robust_acc = 0
            val_n = 0
            for i, batch in enumerate(val_batches):
                X, y = batch['input'], batch['target']

                # Random initialization
                if args.attack == 'none':
                    delta = torch.zeros_like(X)
                else:
                    delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters_test, args.restarts, args.norm, early_stop=args.eval)
                delta = delta.detach()

                robust_output = model(normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
                robust_loss = criterion(robust_output, y)

                output = model(normalize(X))
                loss = criterion(output, y)

                val_robust_loss += robust_loss.item() * y.size(0)
                val_robust_acc += (robust_output.max(1)[1] == y).sum().item()
                val_loss += loss.item() * y.size(0)
                val_acc += (output.max(1)[1] == y).sum().item()
                val_n += y.size(0)

        if not args.eval:
            logger.info('%d \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t %.4f \t %.4f \t \t %.4f \t \t %.4f \t %.4f \t %.4f \t \t %.4f',
                epoch, train_time - start_time, test_time - train_time, lr,
                train_loss/train_n, train_acc/train_n, train_robust_loss/train_n, train_robust_acc/train_n,
                test_loss/test_n, test_acc/test_n, test_robust_loss/test_n, test_robust_acc/test_n)

            if args.val:
                logger.info('validation %.4f \t %.4f \t %.4f \t %.4f',
                    val_loss/val_n, val_acc/val_n, val_robust_loss/val_n, val_robust_acc/val_n)

                if val_robust_acc/val_n > best_val_robust_acc:
                    torch.save({
                            'state_dict':model.state_dict(),
                            'test_robust_acc':test_robust_acc/test_n,
                            'test_robust_loss':test_robust_loss/test_n,
                            'test_loss':test_loss/test_n,
                            'test_acc':test_acc/test_n,
                            'val_robust_acc':val_robust_acc/val_n,
                            'val_robust_loss':val_robust_loss/val_n,
                            'val_loss':val_loss/val_n,
                            'val_acc':val_acc/val_n,
                        }, os.path.join(out_dir, f'model_val.pth'))
                    best_val_robust_acc = val_robust_acc/val_n

            # save checkpoint
            if (epoch+1) % args.chkpt_iters == 0 or epoch+1 == epochs:
                torch.save(model.state_dict(), os.path.join(out_dir,'model_.pth'))
                torch.save(opt.state_dict(), os.path.join(out_dir, 'opt.pth'))

            # save best
            if test_robust_acc/test_n > best_test_robust_acc:
                torch.save({
                        'state_dict':model.state_dict(),
                        'test_robust_acc':test_robust_acc/test_n,
                        'test_robust_loss':test_robust_loss/test_n,
                        'test_loss':test_loss/test_n,
                        'test_acc':test_acc/test_n,
                    }, os.path.join(out_dir, f'model_best.pth'))
                best_test_robust_acc = test_robust_acc/test_n
        else:
            logger.info('%d \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t %.4f \t %.4f \t \t %.4f \t \t %.4f \t %.4f \t %.4f \t \t %.4f',
                epoch, train_time - start_time, test_time - train_time, -1,
                -1, -1, -1, -1,
                test_loss/test_n, test_acc/test_n, test_robust_loss/test_n, test_robust_acc/test_n)
            return


if __name__ == "__main__":
    main()
