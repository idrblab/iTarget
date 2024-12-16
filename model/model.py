import os
from sqlite3 import DatabaseError
import sys
import psutil
from turtle import forward
from matplotlib.pyplot import axis

import torch as th
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from model.metrics import evaluate,reshape_tf2th,to_categorical
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from pathlib import Path
from joblib import dump, load
import itertools
import time
prj_path = Path(__file__).parent.resolve().parent.resolve().parent.resolve().parent.resolve()


def load_model(params, model_path, in_channels, gpuid=-1):
    model = MultimapCNN(params, in_channels=in_channels)
    model.load_state_dict(th.load(os.path.join(model_path/"model.pth")))
    model.device = th.device('cpu' if gpuid == -1 else f'cuda:{gpuid}')
    model.to(model.device)
    return model
    
def save_model(model, model_path):
    model.to("cpu")
    th.save(model.state_dict(), os.path.join(model_path/"model.pth"))
    print(f"Saved PyTorch Model State to {model_path}")

class MultimapCNN_dataset(Dataset):
    def __init__(self, X, Y):
        super().__init__()
        count = len(Y)
        self.src,  self.trg = [], []
        for i in range(count):
            self.src.append((X[0][i],X[1][i]))
            self.trg.append(Y[i])
    def __getitem__(self, index):
        return self.src[index], self.trg[index]
    def __len__(self):
        return len(self.src)

class EarlyStopping:
    """Early stops the training if validation loss or validation acc doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, monitor = 'loss_val'):
        """
        Args:
            patience (int): How long to wait after last time validation loss or validation acc improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss or validation acc improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self._score = None
        self._model = None
        self.early_stop = False
        # self.monitor_better = np.Inf
        self.delta = delta
        self.monitor = monitor

    def __call__(self, score, model, model_path):
        score = score[self.monitor]
        if self.monitor=='loss_val':
            if self._score is None:
                self._score = np.Inf
                self.save_checkpoint(score, model, model_path)
            elif score > self._score - self.delta:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.save_checkpoint(score, model, model_path)
                self.counter = 0

        else:
            if self._score is None:
                self._score = np.Inf
                self.save_checkpoint(score, model, model_path)
            elif score < self._score + self.delta:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.save_checkpoint(score, model, model_path)
                self.counter = 0

    def save_checkpoint(self, score, model, model_path):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            if self.monitor=='loss_val':
                print(f'Validation loss decreased ({self._score:.6f} --> {score:.6f}) ...')
            elif self.monitor=='acc_val':
                print(f'Validation acc increased ({self._score:.6f} --> {score:.6f}) ...')
            elif self.monitor=='auc_val':
                print(f'Validation auc increased ({self._score:.6f} --> {score:.6f}) ...')
            elif self.monitor=='aupr_val':
                print(f'Validation aupr increased ({self._score:.6f} --> {score:.6f}) ...')
            elif self.monitor=='mcc_val':
                print(f'Validation mcc increased ({self._score:.6f} --> {score:.6f}) ...')
            elif self.monitor=='f1_val':
                print(f'Validation f1 increased ({self._score:.6f} --> {score:.6f}) ...')
            elif self.monitor=='recall_val':
                print(f'Validation recall increased ({self._score:.6f} --> {score:.6f}) ...')
            elif self.monitor=='precision_val':
                print(f'Validation precision increased ({self._score:.6f} --> {score:.6f}) ...')
            elif self.monitor=='specificity_val':
                print(f'Validation specificity increased ({self._score:.6f} --> {score:.6f}) ...')
        save_model(model, model_path)
        # self._model = deepcopy(model)
        self._score = score

class MultimapCNN(nn.Module):
    def __init__(self, params, in_channels=None):
        super(MultimapCNN, self).__init__()

        self.random_state = params.random_seed
        self.device = th.device('cpu' if params.gpu == -1 else f'cuda:{params.gpu}')
        self.lr = params.lr
        self.conv1_kernel_size = (9, 13)
        self.dense_layers = [512, 256, 64]
        self.dense_avf = F.relu
        self.last_avf = F.softmax
        self.batch_size = params.batch_size
        self.monitor = params.monitor
        self.metric = params.metric
        self.num_outputs = 2
        self.captum = False

        self.conv1_d = nn.Conv2d(in_channels=in_channels[0], out_channels=48, kernel_size=self.conv1_kernel_size[0], stride = 1, padding = int((self.conv1_kernel_size[0]-1)/2), )
        self.pool1_d = nn.MaxPool2d(kernel_size=3, stride = 2, padding = int((3-1)/2))

        self.Incep1conv1_d = nn.Conv2d(in_channels=48, out_channels=32, kernel_size=5, stride = 1, padding=int((5-1)/2))
        self.Incep1conv2_d = nn.Conv2d(in_channels=48, out_channels=32, kernel_size=3, stride = 1, padding=int((3-1)/2))
        self.Incep1conv3_d = nn.Conv2d(in_channels=48, out_channels=32, kernel_size=1, stride = 1, padding=int((1-1)/2))
        self.pool2_d = nn.MaxPool2d(kernel_size=3, stride = 2, padding = int((3-1)/2))
        self.Incep2conv1_d = nn.Conv2d(in_channels=96, out_channels=64, kernel_size=5, stride = 1, padding=int((5-1)/2))
        self.Incep2conv2_d = nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride = 1, padding=int((3-1)/2))
        self.Incep2conv3_d = nn.Conv2d(in_channels=96, out_channels=64, kernel_size=1, stride = 1, padding=int((1-1)/2))

        self.conv1_p = nn.Conv2d(in_channels=in_channels[1], out_channels=72, kernel_size=self.conv1_kernel_size[1], stride = 1, padding = int((self.conv1_kernel_size[1]-1)/2, ))
        self.pool1_p = nn.MaxPool2d(kernel_size=5, stride = 2, padding = int((5-1)/2))
        self.Incep1conv1_p = nn.Conv2d(in_channels=72, out_channels=48, kernel_size=9, stride = 1, padding=int((9-1)/2))
        self.Incep1conv2_p = nn.Conv2d(in_channels=72, out_channels=48, kernel_size=5, stride = 1, padding=int((5-1)/2))
        self.Incep1conv3_p = nn.Conv2d(in_channels=72, out_channels=48, kernel_size=1, stride = 1, padding=int((1-1)/2))
        self.pool2_p = nn.MaxPool2d(kernel_size=5, stride = 2, padding = int((5-1)/2))
        self.Incep2conv1_p = nn.Conv2d(in_channels=144, out_channels=96, kernel_size=9, stride = 1, padding=int((9-1)/2))
        self.Incep2conv2_p = nn.Conv2d(in_channels=144, out_channels=96, kernel_size=5, stride = 1, padding=int((5-1)/2))
        self.Incep2conv3_p = nn.Conv2d(in_channels=144, out_channels=96, kernel_size=1, stride = 1, padding=int((1-1)/2))

        self.dense1 = nn.Linear(in_features=int((64+64+64)+(96+96+96)), out_features=self.dense_layers[0])
        self.dense2 = nn.Linear(in_features=self.dense_layers[0], out_features=self.dense_layers[1])
        self.dense3 = nn.Linear(in_features=self.dense_layers[1], out_features=self.dense_layers[2])

        self.last = nn.Linear(in_features=self.dense_layers[2], out_features=self.num_outputs)
        

    def Inception1d(self, inputs):
        x1 = F.relu(self.Incep1conv1_d(inputs))
        x2 = F.relu(self.Incep1conv2_d(inputs))
        x3 = F.relu(self.Incep1conv3_d(inputs))
        outputs = th.cat((x1, x2, x3),1)
        return outputs

    def Inception2d(self, inputs):
        x1 = F.relu(self.Incep2conv1_d(inputs))
        x2 = F.relu(self.Incep2conv2_d(inputs))
        x3 = F.relu(self.Incep2conv3_d(inputs))
        outputs = th.cat((x1, x2, x3),1)
        return outputs

    def Inception1c(self, inputs):
        x1 = F.relu(self.Incep1conv1_p(inputs))
        x2 = F.relu(self.Incep1conv2_p(inputs))
        x3 = F.relu(self.Incep1conv3_p(inputs))
        outputs = th.cat((x1, x2, x3),1)
        return outputs

    def Inception2c(self, inputs):
        x1 = F.relu(self.Incep2conv1_p(inputs))
        x2 = F.relu(self.Incep2conv2_p(inputs))
        x3 = F.relu(self.Incep2conv3_p(inputs))
        outputs = th.cat((x1, x2, x3),1)
        return outputs

    def Inception3c(self, inputs):
        x1 = F.relu(self.Incep3conv1_p(inputs))
        x2 = F.relu(self.Incep3conv2_p(inputs))
        x3 = F.relu(self.Incep3conv3_p(inputs))
        outputs = th.cat((x1, x2, x3),1)
        return outputs

    def forward(self, drug_inputs, comp_inputs):
        ## first inputs
        d_conv1 = F.relu(self.conv1_d(drug_inputs))
        d_pool1 = self.pool1_d(d_conv1)
        d_incept1 = self.Inception1d(d_pool1)
        d_pool2 = self.pool2_d(d_incept1) #p2
        d_incept2 = self.Inception2d(d_pool2)

        ## second inputs
        c_conv1 = F.relu(self.conv1_p(comp_inputs))
        c_pool1 = self.pool1_p(c_conv1)
        c_incept1 = self.Inception1c(c_pool1)
        c_pool2 = self.pool2_p(c_incept1) #p2
        c_incept2 = self.Inception2c(c_pool2)

        d_flat1 = F.max_pool2d(input=d_incept2, kernel_size=d_incept2.size()[2:]).squeeze(-1).squeeze(-1) # global pooling
        c_flat1 = F.max_pool2d(input=c_incept2, kernel_size=c_incept2.size()[2:]).squeeze(-1).squeeze(-1) # global pooling

        ## concat
        x = th.cat((d_flat1, c_flat1),1)

        if self.captum:
            return x
        else:
            return x, d_flat1, c_flat1
        
    def get_logits(self, drug_inputs, comp_inputs):
        
        x, d_flat1, c_flat1 = self.forward(drug_inputs, comp_inputs)
 
        ## dense layer
        x = self.dense_avf(self.dense1(x))
        x = self.dense_avf(self.dense2(x))
        x = self.dense_avf(self.dense3(x))


        ## last layer
        outputs = self.last(x) # dont need activation function
        
        return outputs


    def ContrastiveLoss(self, sum, label):
        num = len(sum)

        combinations = list(itertools.combinations(range(num), 2))


        indices = th.tensor(combinations, dtype=th.long).to(self.device)
        sum1 = sum[indices[:, 0]]
        sum2 = sum[indices[:, 1]]
        label1 = label[indices[:, 0]]
        label2 = label[indices[:, 1]]

        contras_label = label1 & ~label2

        cos_distance = 1 - F.cosine_similarity(sum1, sum2, dim=-1)
        loss_contrastive = th.mean((1 - contras_label) * th.pow(cos_distance, 2) + 
                                        (contras_label) * th.pow(th.clamp(2 - cos_distance, min=0.0), 3))
        return loss_contrastive


    def train_loop(self, dataloader, loss_fn, optimizer):
        self.to(self.device)
        # print(f"Using {self.device} device")
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        self.train()

        train_loss= 0
        train_logits,train_label = [],[]
        optimizer.zero_grad()
        for batch, (X, y) in enumerate(dataloader):
            X, y = [x.to(self.device) for x in X], y.to(self.device)
            # Compute prediction error
            embed, _d, _p = self(X[0],X[1])
            loss_contrast = self.ContrastiveLoss(embed, y)

            pred = self.get_logits(X[0],X[1])
            
            loss_label = loss_fn(pred, y)

            total_loss = loss_contrast + loss_label

            # Backpropagation
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # loss_value = loss.item()
            loss_value = total_loss.item()
            if batch % int(num_batches/10) == 0:
                current = batch * len(y)
                print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")

            train_loss += loss_value
            train_logits.append(pred)
            train_label.append(y)

        train_loss /= (batch+1)
        train_logits=th.cat(train_logits, dim=0)
        train_label=th.cat(train_label, dim=0)

        return train_loss, train_logits.cpu().detach(), train_label.cpu().detach()

    def test_loop(self, dataloader, loss_fn, ):
        self.to(self.device)
        # print(f"Using {self.device} device")
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        self.eval()

        test_loss, correct = 0, 0
        test_logits, test_label = [], []
        with th.no_grad():
            for batch, (X, y) in enumerate(dataloader):
            # for X, y in dataloader:
                X, y = [x.to(self.device) for x in X], y.to(self.device)
                embed, _d, _p = self(X[0],X[1])
                pred = self.get_logits(X[0],X[1])
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(th.float).sum().item()
                test_logits.append(pred)
                test_label.append(y)

        test_loss /= num_batches # num_batches==(batch+1)
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        test_logits=th.cat(test_logits, dim=0)
        test_label=th.cat(test_label, dim=0)

        return test_loss, test_logits.cpu(), test_label.cpu()

    def run_loop(self, X, batch_size = 1):
        data = MultimapCNN_dataset(X,np.zeros(len(X[0])))
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
        self.to(self.device)
        # print(f"Using {self.device} device")
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        self.eval()
        run_logits = []
        latents_d = []
        latents_p = []
        with th.no_grad():
            for X, y in dataloader:
                X = [x.to(self.device) for x in X]
                embed, _d, _p = self(X[0],X[1])
                pred = self.get_logits(X[0],X[1])
                run_logits.append(pred)
                latents_d.append(_d)
                latents_p.append(_p)
        run_logits=th.cat(run_logits, dim=0).cpu()
        latents_d=th.cat(latents_d, dim=0).cpu().detach().numpy()
        latents_p=th.cat(latents_p, dim=0).cpu().detach().numpy()
        return run_logits, latents_d, latents_p

