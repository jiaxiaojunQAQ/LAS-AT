# LAS_AT
Code for LAS-AT: Adversarial Training with Learnable Attack Strategy (CVPR2022)

## Introduction
>Adversarial training (AT) is always formulated as a minimax problem, of which the performance depends on the inner optimization that involves the generation of adversarial examples (AEs). Most previous methods adopt Projected Gradient Decent (PGD) with manually specifying attack parameters for AE generation. A combination of the attack parameters can be referred to as an attack strategy. Several works have revealed that using a fixed attack strategy to generate AEs during the whole training phase limits the model robustness and propose to exploit different attack strategies at different training stages to improve robustness. But those multi-stage hand-crafted attack strategies need much domain expertise, and the robustness improvement is limited. In this paper, we propose a novel framework for adversarial training by introducing the concept of “learnable attack strategy”, dubbed LAS-AT, which learns to automatically produce attack strategies to improve the model robustness. Our framework is composed of a target network that uses AEs for training to improve robustness, and a strategy network that produces attack strategies to control the AE generation. Experimental evaluations on three benchmark databases demonstrate the superiority of the proposed method.
## Requirements
Python3 </br>
Pytorch </br>
## Test
> python3.6 test_CIFAR10.py --model_path model.pth --out_dir ./output/ --data-dir cifar-data
> python3.6 test_CIFAR100.py --model_path model.pth --out_dir ./output/ --data-dir cifar-data100
> python3.6 test_TinyImageNet.py --model_path model.pth --out_dir ./output/ --data-dir TinyImageNet
## Trained Models
> The Trained models can be downloaded from the [Baidu Cloud](https://pan.baidu.com/s/1ZEv-7gSEI4gi64PvCnM3ww)(Extraction: 1234.) or the [Google Drive](https://drive.google.com/drive/folders/1972Yhxte4318qbpllyul5dVmvo-VpWVW?usp=sharing)

