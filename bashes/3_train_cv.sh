#!/usr/bin/bash

python -u ../main.py --kfold_num 5 --task cv --n_epochs 128 --gpu 0 --batch_size 512 --lr 5e-4 --monitor auc_val --source example
