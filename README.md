# LAS_AT
Code for LAS-AT: Adversarial Training with Learnable Attack Strategy (CVPR2022 Oral)

## Introduction
>Adversarial training (AT) is always formulated as a minimax problem, of which the performance depends on the inner optimization that involves the generation of adversarial examples (AEs). Most previous methods adopt Projected Gradient Decent (PGD) with manually specifying attack parameters for AE generation. A combination of the attack parameters can be referred to as an attack strategy. Several works have revealed that using a fixed attack strategy to generate AEs during the whole training phase limits the model robustness and propose to exploit different attack strategies at different training stages to improve robustness. But those multi-stage hand-crafted attack strategies need much domain expertise, and the robustness improvement is limited. In this paper, we propose a novel framework for adversarial training by introducing the concept of “learnable attack strategy”, dubbed LAS-AT, which learns to automatically produce attack strategies to improve the model robustness. Our framework is composed of a target network that uses AEs for training to improve robustness, and a strategy network that produces attack strategies to control the AE generation. Experimental evaluations on three benchmark databases demonstrate the superiority of the proposed method.
## Requirements
Python3 </br>
Pytorch </br>

## Train for LAS-PGD-AT
* On CIFAR10
> python3 LAS_AT_train_cifar10.py --model WideResNet --epsilon_types 3 4 5 6 7 8 9 10 11 12 13 14 15 --attack_iters_types 3 4 5 6 7 8 9 10 11 12 13 14 --step_size_types  1 2 3 4 5  --epochs 110  --data-dir cifar-data  --out-dir CIFAR10/LAS_PGD_AT 
* On CIFAR100
> python3 LAS_AT_train_cifar100.py --model WideResNet --epsilon_types 3 4 5 6 7 8 9 10 11 12 13 14 15 --attack_iters_types 3 4 5 6 7 8 9 10 11 12 --step_size_types  1 2 3 4 5  --epochs 110  --data-dir cifar-data100  --out-dir CIFAR100/LAS_PGD_AT
* On TinyImageNet
> python3 LAS_AT_train_TinyImageNet.py --model PreActResNest18 --epsilon_types 3 4 5 6 7 8 9 10 11 12 13 14 15 --attack_iters_types 3 4 5 6 7 8 9 10 11 12 13 14 --step_size_types  1 2 3 4 5  --epochs 110  --data-dir tiny-imagenet-200  --out-dir TinyImageNet/LAS_PGD_AT

## Train for LAS-Trades
* On CIFAR10
> python3 LAS_Trades_train_cifar10.py --model WideResNet --epsilon_types  5 6 7 8 9  --attack_iters_types  7 8 9 10 11 12 13 14 --step_size_types  2 3 4 --beta_types 5 6 7 8 9  --epochs 100  --data-dir cifar-data  --out-dir CIFAR10/LAS_Trades 
* On CIFAR100
> python3 LAS_Trades_train_cifar100.py --model WideResNet --epsilon_types 5 6 7 8 9 10 --attack_iters_types 7 8 9 10 11 12 13 14 --step_size_types  2 3 4  --beta_types 5 6 7 8 9  --epochs 100 --data-dir cifar-data100  --out-dir CIFAR100/LAS_Trades
* On TinyImageNet
> python3 LAS_Trades_train_TinyImageNet.py --model PreActResNest18 --epsilon_types 5 6 7 8 9 10 --attack_iters_types 7 8 9 10 11 12 13 14 --step_size_types  2 3 4 --beta_types 5 6 7 8 9  --epochs 110  --data-dir tiny-imagenet-200  --out-dir TinyImageNet/LAS_Trades

## Train for LAS-AWP
* On CIFAR10
> python3 LAS_AWP_train_cifar10.py --model WideResNet --epsilon_types 7 8 9 10 11 12 13 14 15 --attack_iters_types 8 9 10 11 12 13 14 15 16 --step_size_types 2 3 4 5  --epochs 200  --data-dir cifar-data  --out-dir CIFAR10/LAS_AWP 
* On CIFAR100
> python3 LAS_AWP_train_cifar100.py --model WideResNet --epsilon_types 7 8 9 10 11 12 13 14 15 --attack_iters_types 8 9 10 11 12 13 14 15 --step_size_types  2 3 4 5  --epochs 200  --data-dir cifar-data100  --out-dir CIFAR100/LAS_AWP
* On TinyImageNet
> python3 LAS_AWP_train_TinyImageNet.py --model PreActResNest18 --epsilon_types 7 8 9 10 11 12 13 14 15 --attack_iters_types 8 9 10 11 12 13 14 15 --step_size_types 2 3 4 5  --epochs 200  --data-dir tiny-imagenet-200  --out-dir TinyImageNet/LAS_AWP




## Test
> + python3.6 test_CIFAR10.py --model_path model.pth --out_dir ./output/ --data-dir cifar-data </br>
> + python3.6 test_CIFAR100.py --model_path model.pth --out_dir ./output/ --data-dir cifar-data100 </br>
> + python3.6 test_TinyImageNet.py --model_path model.pth --out_dir ./output/ --data-dir tiny-imagenet-200
## Trained Models
> The Trained models can be downloaded from the [Baidu Cloud](https://pan.baidu.com/s/1fmnO9jZw5Fcwy5B28bvRSw)(Extraction: 1234.) or the [Google Drive](https://drive.google.com/drive/folders/13ZZGAIzXuvfCvjMGD69Qude3p3gE0r7b?usp=sharing)
