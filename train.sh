#!/bin/bash
# python train_ddag.py --dataset regdb --lr 0.1 --graph --wpa --part 3 --gpu 1
python train_ddag.py --dataset sysu --lr 0.01 --graph --wpa --part 3 --gpu 0 --nheads 4 --epochs 100

# python train_ddag.py --dataset sysu --lr 0.01 --graph --wpa --part 3 --gpu 0 --nheads 8 --resume "sysu_G_nh8_P_3_drop_0.2_4_8_lr_0.01_best.t" 
