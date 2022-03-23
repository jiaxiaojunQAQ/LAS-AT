# encoding: utf-8
import argparse
from utils import *


import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Bernoulli
import copy
import os
import numpy
import argparse
import logging
from StrategyNet import *
from torch.nn.utils import clip_grad_norm_
logger = logging.getLogger(__name__)
CUDA_LAUNCH_BLOCKING=1
from tensorboardX import SummaryWriter
from CIFAR100_models import *

def get_args():
    parser = argparse.ArgumentParser('LAS_Trades')
    # target model
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='../cifar-data100', type=str)
    parser.add_argument('--out-dir', default='LAS_Trades', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--target_model_lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--target_model_lr_scheduler', default='multistep', type=str, choices=['cyclic', 'multistep'])
    parser.add_argument('--target_model_lr_min', default=0., type=float)
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--model', default='WideResNet', type=str, help='model name')
    parser.add_argument('--weight-decay', '--wd', default=5e-4,
                        type=float, metavar='W')


    parser.add_argument('--path', default='LAS_Trades', type=str, help='model name')

    ## search
    parser.add_argument('--attack_types', type=list, default=['IFGSM'], help='all searched policies')
    parser.add_argument('--epsilon_types', type=int, nargs="*", default=range(1, 11))
    parser.add_argument('--attack_iters_types', type=int, nargs="*", default=range(5, 10))
    parser.add_argument('--step_size_types', type=int, nargs="*", default=range(2, 4))
    parser.add_argument('--beta_types', type=int, nargs="*", default=range(5, 9))
    ## policy Hyperparameters
    parser.add_argument('--policy_model_lr', type=float, default=0.0001)
    parser.add_argument('--policy_model_lr_scheduler', default='multistep', type=str, choices=['cyclic', 'multistep'])
    parser.add_argument('--policy_model_lr_min', default=0., type=float)
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')

    print('**')

    parser.add_argument('--interval_num', type=int, default=1)
    parser.add_argument('--exp_iter', type=int, default=1)

    parser.add_argument('--tensor-path', default='runs', type=str, help='tensorboardX name')

    parser.add_argument('--policy_optimizer', default='SGD_with_momentum', type=str, help='policy_optimizer')
    parser.add_argument('--a', default=1, type=float)
    parser.add_argument('--b', default=1, type=float)
    parser.add_argument('--c', default=1, type=float)


    parser.add_argument('--R2_param', default=8, type=float)
    parser.add_argument('--R3_param', default=2, type=float)
    parser.add_argument('--clip_grad_norm', default=1.0, type=float)




    arguments = parser.parse_args()
    return arguments


args = get_args()

out_dir=os.path.join(args.out_dir,'model_'+args.model)

out_dir=os.path.join(out_dir,'epsilon_types_'+str(min(args.epsilon_types))+'_'+str(max(args.epsilon_types)))
out_dir=os.path.join(out_dir,'attack_iters_types_'+str(min(args.attack_iters_types))+'_'+str(max(args.attack_iters_types)))
out_dir=os.path.join(out_dir,'step_size_types_'+str(min(args.step_size_types))+'_'+str(max(args.step_size_types)))
out_dir=os.path.join(out_dir,'beta_types_'+str(min(args.beta_types))+'_'+str(max(args.beta_types)))





tensor_path=os.path.join(out_dir,'runs')
# writer = SummaryWriter(tensor_path)

eps = np.finfo(np.float32).eps.item()
criterion_kl = nn.KLDivLoss(size_average=False)



print(out_dir)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
logfile = os.path.join(out_dir, 'output.log')
if os.path.exists(logfile):
    os.remove(logfile)
logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
    filename=logfile)
logger.info(args)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_loader, test_loader = get_loaders_cifar100(args.data_dir, args.batch_size)
best_acc = 0
best_clean_acc=0
start_epoch = 0





def _get_all_policies(attack_id_list, espilon_id_list, attack_iters_id_list, step_size_id_list,beta_id_list, args):
    policies = []
    #print(attack_id_list)
    attack_id_list = attack_id_list[0].cpu().numpy()
    espilon_id_list = espilon_id_list[0].cpu().numpy()
    attack_iters_id_list = attack_iters_id_list[0].cpu().numpy()
    step_size_id_list = step_size_id_list[0].cpu().numpy()
    beta_id_list = beta_id_list[0].cpu().numpy()
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
        all_policy['beta'] = args.beta_types[beta_id_list[n]]

        # all_policy['prob'] = args.prob_types[prob_id_list[n]]
        sub_policy[n] = all_policy
        policies.append(sub_policy)

    return policies





def select_action(policy_model, state):
    # policy_model = policy_model.eval()

    outputs = policy_model(state)
    attack_id_list = []
    espilon_id_list = []
    attack_iters_id_list = []
    step_size_id_list = []
    beta_id_list=[]
    prob_list = []
    action_list = []

    max_attack_id_list = []
    max_espilon_id_list = []
    max_attack_iters_id_list = []
    max_step_size_id_list = []
    max_beta_id_list=[]
    # max_prob_list = []
    # max_action_list = []
    temp_saved_log_probs = []
    for id in range(5):

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
        elif mode == 4:
            beta_id_list.append(action)
            max_beta_id_list.append(max_action)
        temp_saved_log_probs.append(m.log_prob(action))
    #policy_model.saved_log_probs.append(temp_saved_log_probs)
    curpolicy = _get_all_policies(attack_id_list, espilon_id_list, attack_iters_id_list, step_size_id_list,beta_id_list, args)
    max_curpolicy = _get_all_policies(max_attack_id_list, max_espilon_id_list, max_attack_iters_id_list,
                                      max_step_size_id_list,max_beta_id_list, args)
    action_list.append(attack_id_list)
    action_list.append(espilon_id_list)
    action_list.append(attack_iters_id_list)
    action_list.append(step_size_id_list)
    action_list.append(beta_id_list)

    return action_list, curpolicy, prob_list, max_curpolicy


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X + delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta



def train_target_model(input_batch,delta,y_batch, copy_target_model,beta_batch):
    X, Y = input_batch.to(device), y_batch.to(device)
    delta=delta.to(device)
    batch_size = len(X)
    criterion_kl_not_sum = nn.KLDivLoss(size_average=False, reduce=False)
    target_lr = target_model_scheduler.get_lr()[0]
    optimizer = optim.SGD(copy_target_model.parameters(), lr=target_lr, momentum=0.9, weight_decay=args.weight_decay)
    copy_target_model.train()
    optimizer.zero_grad()
    clean_outputs = copy_target_model(X)
    #loss = criterion(outputs, Y)

    loss_natural = F.cross_entropy(clean_outputs, Y)
    all_loss_robust = beta_batch * torch.sum(
        criterion_kl_not_sum(F.log_softmax(copy_target_model(X + delta), dim=1), F.softmax(copy_target_model(X), dim=1)), dim=1)
    #print(all_loss_robust.shape)
    loss_robust = (1.0 / batch_size) * torch.sum(all_loss_robust, dim=0)
    # loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(copy_target_model(X + delta), dim=1),
    #                                                          F.softmax(copy_target_model(X), dim=1))
    loss = loss_natural + loss_robust
    loss.backward()
    optimizer.step()
    return copy_target_model


def Attack_policy(input_batch, y_batch, target_model, policies):
    target_model.eval()
    criterion = nn.CrossEntropyLoss()
    X, y = input_batch.cuda(), y_batch.cuda()
    delta = torch.zeros_like(X).cuda()
    delta.requires_grad = True
    for ii in range(len(policies)):
        epsilon = (policies[ii][ii]['epsilon'] / 255.) / std
        alpha = (policies[ii][ii]['step_size'] / 255.) / std

        temp_X = X[ii:ii + 1]
        temp_delta = torch.zeros_like(temp_X).cuda()
        temp_delta.requires_grad = True
        for _ in range(policies[ii][ii]['attack_iters']):
            # print((temp_X + temp_delta).shape)
            output = target_model(temp_X + temp_delta)
            loss = criterion(output, y[ii:ii + 1])
            # print(loss)
            loss.backward()
            grad = temp_delta.grad.detach()

            temp_delta.data = clamp(temp_delta + alpha * torch.sign(grad), -epsilon, epsilon)
            temp_delta.data = clamp(temp_delta, lower_limit - temp_X, upper_limit - temp_X)
            temp_delta.grad.zero_()
        temp_delta = temp_delta.detach()
        delta[ii:ii + 1] = temp_delta
    delta = delta.detach()
    target_model.train()
    return delta


def Attack_policy_batch(input_batch, y_batch, target_model, policies):
    target_model.eval()
    X, y = input_batch.cuda(), y_batch.cuda()
    delta_batch = torch.zeros_like(X).cuda()
    delta_batch.requires_grad = True
    alpha_batch = []
    epsilon_batch = []
    attack_iters_batch = []
    beta_batch=[]
    criterion_kl = nn.KLDivLoss(size_average=False)
    for ii in range(len(policies)):
        epsilon = (policies[ii][ii]['epsilon'] / 255.) / std
        epsilon_batch.append(epsilon.cpu().numpy())

        alpha = (policies[ii][ii]['step_size'] / 255.) / std
        alpha_batch.append(alpha.cpu().numpy())

        attack_iters = policies[ii][ii]['attack_iters']
        temp_batch = torch.randint(attack_iters, attack_iters + 1, (3, 1, 1))
        attack_iters_batch.append(temp_batch.cpu().numpy())
        # print(attack_iters)
        beta = policies[ii][ii]['beta']
        # print(beta)
        beta_batch.append(beta)


    alpha_batch = torch.from_numpy(numpy.array(alpha_batch)).cuda()
    epsilon_batch = torch.from_numpy(numpy.array(epsilon_batch)).cuda()
    attack_iters_batch = torch.from_numpy(numpy.array(attack_iters_batch)).cuda()
    beta_batch = torch.from_numpy(numpy.array(beta_batch)).cuda()

    max_attack_iters = torch.max(attack_iters_batch).cpu().numpy()
    # print(torch.max(attack_iters_batch))
    x_adv = X.detach() + 0.001 * torch.randn(X.shape).cuda().detach()
    for _ in range(max_attack_iters):
        mask_bacth = attack_iters_batch.ge(1).float()
        # print(alpha_batch[0])
        x_adv.requires_grad_()
        with torch.enable_grad():
            adv_kl = F.log_softmax(target_model(x_adv), dim=1)
            clean_kl = F.softmax(target_model(X), dim=1)
            loss_kl = criterion_kl(adv_kl, clean_kl)

        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + mask_bacth * alpha_batch * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, X - epsilon_batch), X + epsilon_batch)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

        attack_iters_batch = attack_iters_batch - 1

    # print( lower_limit.shape)
    # print( torch.sign(grad).shape)
    delta_batch = x_adv - X
    delta_batch = delta_batch.detach()

    target_model.train()
    return delta_batch,beta_batch

def Get_delta(input_batch, y_batch, target_model, action):
    target_model.eval()
    inputs, targets = input_batch.to(device), y_batch.to(device)
    delta,beta_batch = Attack_policy_batch(input_batch, y_batch, target_model, action)
    target_model.train()

    return inputs + delta,beta_batch



def Get_reward(input_batch, y_batch, target_model, action):


    criterion_kl_not_sum = nn.KLDivLoss(size_average=False, reduce=False)
    #criterion_kl_sum= nn.KLDivLoss(size_average=False, reduce=True)
    inputs, targets = input_batch.to(device), y_batch.to(device)
    batch_size = len(inputs)
    delta,beta_batch = Attack_policy_batch(input_batch, y_batch, target_model, action)
    print(beta_batch)
    with torch.no_grad():
        ori_clean_output = target_model(inputs)
        output = target_model(inputs + delta)
    # logsoftmax_func = nn.LogSoftmax(dim=1)
    # soft_output = logsoftmax_func(output)
    # y_one_hot = F.one_hot(y_batch, 10).float()
    # print(y_one_hot.shape)
    loss_natural = F.cross_entropy(ori_clean_output, targets)
    all_loss_robust=beta_batch*torch.sum(criterion_kl_not_sum(F.log_softmax(output, dim=1),F.softmax(ori_clean_output, dim=1)),dim=1)
    #print(all_loss_robust.shape)
    loss_robust = (1.0 / batch_size) *torch.sum( all_loss_robust,dim=0)#### R1的奖励函数

    #mean_loss_robust=args.beta  *criterion_kl_sum(F.log_softmax(output, dim=1),F.softmax(ori_clean_output, dim=1))

    #print('loss_robust',loss_robust)
    #print('mean_loss_robust', mean_loss_robust)
    R1=loss_natural +  loss_robust
    R1 = torch.clamp(R1, 0, 20)


    copy_target_model = copy.deepcopy(target_model)
    copy_target_model.train()
    copy_target_model = train_target_model(inputs, delta, targets, copy_target_model,beta_batch)
    epsilon = (8 / 255.) / std
    alpha = (2 / 255.) / std
    pgd_delta = attack_pgd(copy_target_model, inputs, targets, epsilon, alpha, 10, 2)
    copy_target_model.eval()
    with torch.no_grad():
        R2_output = copy_target_model(inputs + pgd_delta)
        clean_output = copy_target_model(inputs)
    # # logsoftmax_func = nn.LogSoftmax(dim=1)
    # # soft_output = logsoftmax_func(output)
    # # y_one_hot = F.one_hot(y_batch, 10).float()
    # # print(y_one_hot.shape)
    R2 = (R2_output.max(1)[1] == targets).sum().item()
    R3 = (clean_output.max(1)[1] == targets).sum().item()

    test_n = targets.size(0)
    R2 = (R2) / test_n * args.R2_param
    R3 = (R3) / test_n * args.R3_param

    R2 = torch.clamp(torch.tensor(R2), -20, 20)
    R3 = torch.clamp(torch.tensor(R3), -20, 20)
    print('R1:', R1)

    print("R2:", R2)

    print("R3:", R3)
    return (args.a * R1 + args.b * R2 + args.c * R3), R1, R2, R3, inputs + delta


Strategy_model = ResNet18_Strategy(args)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
Strategy_model.cuda()
Strategy_model.train()
print(Strategy_model)
#policy_optimizer = optim.SGD(Policy_model.parameters(), lr=args.policy_model_lr)
if args.policy_optimizer=='SGD_without_momentum':
    policy_optimizer = optim.SGD(Strategy_model.parameters(), lr=args.policy_model_lr)
elif args.policy_optimizer=='SGD_with_momentum':
    policy_optimizer = optim.SGD([{'params': Strategy_model.parameters(), 'initial_lr': args.policy_model_lr}], lr=args.policy_model_lr,momentum=0.9, weight_decay=5e-4)
elif args.policy_optimizer == "Adam_with_momentum":
    policy_optimizer=torch.optim.Adam(Strategy_model.parameters(),lr=args.policy_model_lr,weight_decay=5e-4)
# 所用攻击方式的个数  攻击强度 迭代次数 步长

print('==> Building model..')
logger.info('==> Building model..')


if args.model == "ResNet18":
    target_model = ResNet18()

elif args.model == "WideResNet":
    target_model = WideResNet()
device_id=range(torch.cuda.device_count())
if len(device_id)>1:
    target_model=torch.nn.DataParallel(target_model);
target_model = target_model.to(device)
criterion = nn.CrossEntropyLoss()
target_model_optimizer = optim.SGD(target_model.parameters(), lr=args.target_model_lr, momentum=0.9, weight_decay=args.weight_decay)
lr_steps = args.epochs * len(train_loader)

if args.target_model_lr_scheduler == 'cyclic':
    target_model_scheduler = torch.optim.lr_scheduler.CyclicLR(target_model_optimizer, base_lr=args.target_model_lr_min,
                                                               max_lr=args.target_model_lr,
                                                               step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
elif args.target_model_lr_scheduler == 'multistep':
    target_model_scheduler = torch.optim.lr_scheduler.MultiStepLR(target_model_optimizer,
                                                                  milestones=[int(lr_steps* 75/100), int(lr_steps * 90 / 100)],
                                                                  gamma=0.1)
policy_lr_steps=int(args.epochs * len(train_loader)*args.exp_iter / args.interval_num)
if args.policy_model_lr_scheduler == 'cyclic':
    policy_model_scheduler = torch.optim.lr_scheduler.CyclicLR(policy_optimizer, base_lr=args.policy_model_lr_min,
                                                               max_lr=args.policy_model_lr,
                                                               step_size_up=policy_lr_steps / 2, step_size_down=policy_lr_steps / 2)
elif args.policy_model_lr_scheduler == 'multistep':
    policy_model_scheduler = torch.optim.lr_scheduler.MultiStepLR(policy_optimizer,
                                                                  milestones=[int(lr_steps* 75/100), int(lr_steps * 90 / 100)],
                                                                  gamma=0.1)
target_model_path = os.path.join(out_dir, 'target_model_ckpt.t7')
from collections import OrderedDict
if os.path.exists(target_model_path):
        print("resuming............................................")
        logger.info("resuming............................................")
        #start_epoch = args.resume
        target_model_path = os.path.join(out_dir, 'target_model_ckpt.t7')
        target_model_checkpoint = torch.load(target_model_path)
        start_epoch = target_model_checkpoint['epoch']
        try:
            target_model.load_state_dict(target_model_checkpoint['net'])
        except:
            new_state_dict = OrderedDict()
            for k, v in target_model_checkpoint['net'].items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            target_model.load_state_dict(new_state_dict,False)
        policy_model_path=os.path.join(out_dir,'policy_model_ckpt.t7')
        policy_model_checkpoint = torch.load(policy_model_path)
        try:
            Strategy_model.load_state_dict( policy_model_checkpoint['net'])
        except:
            new_state_dict = OrderedDict()
            for k, v in  policy_model_checkpoint['net'].items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            Strategy_model.load_state_dict(new_state_dict,False)

        target_model_optimizer_path=os.path.join(out_dir, 'target_model_optimizer.pth')
        target_model_optimizer.load_state_dict(torch.load(target_model_optimizer_path))
        #torch.save(policy_optimizer.state_dict(), os.path.join(out_dir, 'policy_model_optimizer.pth'))
        policy_optimizer_path = os.path.join(out_dir, 'policy_model_optimizer.pth')
        policy_optimizer.load_state_dict(torch.load(policy_optimizer_path))

        target_model_scheduler_path = os.path.join(out_dir, 'target_model_scheduler.pth')
        target_model_scheduler.load_state_dict(torch.load(target_model_scheduler_path))

        policy_model_scheduler_path = os.path.join(out_dir, 'policy_model_scheduler.pth')
        policy_model_scheduler.load_state_dict(torch.load(policy_model_scheduler_path))

        # if os.path.exists(os.path.join(args.fname, f'model_best.pth')):
        #     best_test_robust_acc = torch.load(os.path.join(args.fname, f'model_best.pth'))['test_robust_acc']
        # if args.val:
        #     best_val_robust_acc = torch.load(os.path.join(args.fname, f'model_val.pth'))['val_robust_acc']
        best_target_model_path = os.path.join(out_dir, 'best_target_model_ckpt.t7')
        best_target_model_checkpoint = torch.load(best_target_model_path)

        best_acc=best_target_model_checkpoint['best_acc']
        best_clean_acc = best_target_model_checkpoint['best_clean_acc']
        logger.info('Test Acc  \t PGD Acc')
        logger.info('%.4f \t  \t %.4f', best_clean_acc, best_acc)
else:
        start_epoch = 0


global curr_step
curr_step = 0

import time
def train(epoch):
    print('\nEpoch: %d' % epoch)
    # logger.info('\nEpoch: %d' % epoch)
    start_epoch_time = time.time()
    train_loss = 0
    train_acc = 0
    train_n = 0
    logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)


        global curr_step
        curr_step = curr_step + 1
        if curr_step%args.interval_num==0:
            # while num_iter < 1:


            # print(targets.shape)
            # print(Get_reward(inputs,targets,target_model, policies).shape)
            # outputs = net(inputs)
            # print(select_action(net,inputs))
            #####训练policy model
            print("*******************train policy model**********************")
            #logger.info("*******************train policy model**********************")
            # epis_rewards = []
            pocliy_inputs=inputs.clone().cuda()



            Strategy_model.train()

            action_list, policy_outputs, policy_prob, max_policy_outputs = select_action(Strategy_model, pocliy_inputs)

                # print(policy_outputs)a
            reward, R1, R2, R3,adv_examples = Get_reward(pocliy_inputs, targets, target_model, policy_outputs)



                #baseline_reword, R1_baseline, R2_baseline, R3_baseline = Get_reward(inputs, targets, target_model,max_policy_outputs)
                # epis_rewards.append(reward)
            log_probs = []
            policy_loss = []
            for j in range(5):
                log_probs.append(policy_prob[j].log_prob(action_list[j][0]))
                policy_loss.append(-log_probs[j] * reward)
            # print(action_list)
            # logger.info(action_list)
            policy_loss = (policy_loss[0].mean() + policy_loss[1].mean() + policy_loss[2].mean() + policy_loss[3].mean()+policy_loss[4].mean())

            policy_optimizer.zero_grad()
            # torch.nn.utils.clip_grad_norm_(Policy_model.parameters(), 5.0)
            print(policy_loss)
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(Strategy_model.parameters(), args.clip_grad_norm)
            policy_optimizer.step()
            policy_model_scheduler.step()

        #####训练target model
        print("*******************train target model**********************")
        criterion_kl_not_sum = nn.KLDivLoss(size_average=False, reduce=False)
        # outputs = Policy_model(inputs)
        pocliy_inputs1=inputs.clone().cuda()
        for _ in range(args.exp_iter):
            Strategy_model.eval()

            action_list, policy_outputs, policy_prob, max_policy_outputs = select_action(Strategy_model, pocliy_inputs1)

            print(policy_outputs)
            #.info(policy_outputs)
            adv_examples,beta_batch= Get_delta(pocliy_inputs1, targets, target_model, policy_outputs)
            pocliy_inputs1=adv_examples
        # action_list, cur_policies, policy_prob, max_policy_outputs = select_action(Policy_model, inputs)
        # cur_delta = Attack_policy_batch(inputs, targets, target_model, cur_policies)
        # cur_delta = cur_delta.detach()
        target_model.train()
        batch_size=len(inputs)
        logits = target_model(inputs)
        logits_adv=target_model(pocliy_inputs1)
        loss_natural = F.cross_entropy(logits, targets)
        all_loss_robust = beta_batch * torch.sum(
            criterion_kl_not_sum(F.log_softmax(logits_adv, dim=1),
                                 F.softmax(logits, dim=1)), dim=1)
        #print(all_loss_robust.shape)
        loss_robust = (1.0 / batch_size) * torch.sum(all_loss_robust, dim=0)


        target_loss = loss_natural +loss_robust
        #train_acc = (logits.max(1)[1] == targets).sum().item()
        target_model_optimizer.zero_grad()
        target_loss.backward()
        target_model_optimizer.step()
        target_model_scheduler.step()
        train_loss += target_loss.item() * targets.size(0)
        train_acc += (logits_adv.max(1)[1] == targets).sum().item()
        train_n += targets.size(0)
        print("Target model loss:", target_loss)
    epoch_time = time.time()

    lr = target_model_scheduler.get_lr()[0]
    logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f',
                epoch, epoch_time - start_epoch_time, lr, train_loss / train_n, train_acc / train_n)


def test(epoch):
    global best_acc
    global best_clean_acc
    target_model.eval()
    test_loss = 0
    correct = 0
    total = 0
    pgd_loss, pgd_acc = evaluate_pgd(test_loader, target_model, 10, 1)
    test_loss, test_acc = evaluate_standard(test_loader, target_model)
    acc = pgd_acc
    state = {
        'net': target_model.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }

    state1 = {
        'net': Strategy_model.state_dict(),
    }

    target_path = os.path.join(out_dir, 'target_model_ckpt.t7')
    policy_path = os.path.join(out_dir, 'policy_model_ckpt.t7')
    torch.save(state, target_path)
    torch.save(state1, policy_path)
    torch.save(target_model_optimizer.state_dict(), os.path.join(out_dir, 'target_model_optimizer.pth'))
    torch.save(policy_optimizer.state_dict(), os.path.join(out_dir, 'policy_model_optimizer.pth'))

    torch.save(policy_model_scheduler.state_dict(), os.path.join(out_dir, 'policy_model_scheduler.pth'))
    torch.save(target_model_scheduler.state_dict(), os.path.join(out_dir, 'target_model_scheduler.pth'))
    # Save checkpoint.
    # Save checkpoint.

    # print('Test acc:', test_acc)
    # print('Val acc:', acc)
    # logger.info('Test acc: ', test_acc)
    # logger.info('Val acc: ', acc)
    if acc >=best_acc:

        print('Saving..')
        # logger.info("Saving..")
        state = {
            'net': target_model.state_dict(),
            'best_clean_acc':test_acc,
            'best_acc': acc,
            'epoch': epoch,
        }

        state1 = {
            'net': Strategy_model.state_dict(),
        }

        target_path = os.path.join(out_dir, 'best_target_model_ckpt.t7')
        policy_path = os.path.join(out_dir, 'best_policy_model_ckpt.t7')
        torch.save(state, target_path)
        torch.save(state1, policy_path)
        best_acc = acc
        best_clean_acc = test_acc

    print(best_acc)
    # logger.info(best_acc)
    # logger.info(test_acc)
    logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
    logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc)
    logger.info('Test Acc  \t PGD Acc')
    logger.info('%.4f \t  \t %.4f',  best_clean_acc, best_acc)

    return best_acc


for epoch in range(start_epoch,  args.epochs):
    train(epoch)
    print("*****************************************test*************************")
    #logger.info(("*****************************************test*************************"))
    result_acc = test(epoch)
    print(result_acc)
    #logger.info(result_acc)
