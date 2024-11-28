
from run.cross_validation import cross_valid
from run.run import run
from run.prediction import predict
import argparse
import torch as th
import numpy as np
import random
import os

def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	th.manual_seed(seed)
	th.cuda.manual_seed(seed)
	th.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	th.backends.cudnn.benchmark = False
	th.backends.cudnn.deterministic = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=888)
    parser.add_argument("--kfold_num", type=int, default=5, help="number of folds for K-Fold Cross-Validation, default 5")
    parser.add_argument("--kfold_cv_type", type=str, default='pair', choices=['pair', 'drug', 'protein'], help="K-Fold Cross-Validation dataset splitting choice")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU id, -1 for cpu")
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight for L2 loss") 
    parser.add_argument("--n_epochs", type=int, default=512, help="number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="model batch size")
    parser.add_argument("--n_attn_heads", type=int, default=1, help="num of attention heads at layer")
    parser.add_argument("--attn_units", type=int, default=8, help="attention_size at each attention head, its length equals n_attn_heads")
    parser.add_argument("--task", type=str, default='cv', choices=['cv', 'run', 'predict'], help="model work mode, cross-validating, running or predicting")
    parser.add_argument("--monitor", type=str, default='loss_val', choices=['loss_val', 'acc_val', 'auc_val', 'aupr_val', 'mcc_val', 'f1_val', 'recall_val', 'precision_val', 'specificity_val'], help="earlystop monitor")
    parser.add_argument("--metric", type=str, default='ACC', choices=['ACC', 'ROC'], help="optimaizer metric")
    parser.add_argument("--source", type=str, default='bindingdb', help="the data source that the task working on")
    parser.add_argument("--unseen", type=str, default='unseen_drug', help="the test set of unseen or seen drugs or proteins")
    parser.add_argument("--scale_method", type=str, default='standard', choices=['standard', 'minmax', 'None'], help="")
    parser.add_argument("--disttype", type=str, default='ALL', choices=['ALL','bindingdb'], help="the data source that the task working on")



    params = parser.parse_args()
    print(vars(params))

    seed_torch(params.random_seed)

    if params.task=='cv':
        cver = cross_valid(params)
        cver.run()
    elif params.task=='run':
        runner = run(params)
        runner.run()
    elif params.task=='predict':
        predictner = predict(params)
        predictner.run()