import os
import torch
from copy import deepcopy
import numpy as np
import xarray as xr
import pandas as pd
import torch.nn as nn
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import zipfile
import torchvision.models as models
from openstl.models.mfwpn import MFWPN_Model
import torch
import torch.nn as nn
from config import configs
from torch.utils.data import DataLoader
import pickle
import math
from matplotlib.pyplot import MultipleLocator
from utils.data_sliding import *
import pywt
import pywt.data
import torch.nn.functional as F
from utils import SSIM
from thop import profile

class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):

        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

class Trainer:
    def __init__(self, configs):
        self.configs = configs
        self.device = configs.device
        torch.manual_seed(35)
        self.network = MFWPN_Model().to(configs.device)
        adam = torch.optim.Adam([{'params': self.network.parameters()}], lr=0, weight_decay=configs.weight_decay)
        factor = math.sqrt(configs.d_model*configs.warmup)*0.001
        self.opt = NoamOpt(configs.d_model, factor, warmup=configs.warmup, optimizer=adam)
        self.u, self.v = 'u', 'v'

    def loss(self, y_pred, y_true, idx):
        if idx == 'u':
            idx = 0
        if idx == 'v':
            idx = 1
            
        rmse = torch.mean((y_pred[:, :, idx] - y_true[:, :, idx])**2, dim=[2, 3])
        rmse = torch.mean(torch.sqrt(rmse.mean(dim=0)))
            
        return rmse

    def test(self, dataloader_test, ele):
        uv_pred = []
        with torch.no_grad():
            for input_uv, uv_true in dataloader_test:    
                uv = self.network(input_uv.float().to(self.device), ele.float().to(self.device))
                uv_pred.append(uv)

        return torch.cat(uv_pred, dim=0)

    def infer(self, dataset, dataloader, ele):
        self.network.eval()
        with torch.no_grad():
            uv_pred = self.test(dataloader, ele)
            uv_true = torch.from_numpy(dataset.target).float().to(self.device)
            
            uv_pred_np = uv_pred
            uv_true_np = uv_true
            
            uv_pred_test = uv_pred_np.to('cpu')
            uv_true_test = uv_true_np.to('cpu')
            
            uv_pred_test = uv_pred_test.numpy()
            uv_true_test = uv_true_test.numpy()
            
            np.save(file='result/uv_pred', arr=uv_pred_test)
            np.save(file='result/uv_true', arr=uv_true_test)
            
            loss_u = self.loss(uv_pred, uv_true, self.u).item()
            loss_v = self.loss(uv_pred, uv_true, self.v).item()

        return loss_u, loss_v

class dataset_package(Dataset):
    def __init__(self, train_x, train_y):
        super().__init__()
        self.input = train_x
        self.target = train_y

    def GetDataShape(self):
        return {'input': self.input.shape,
                'target': self.target.shape}

    def __len__(self, ):
        return self.input.shape[0]

    def __getitem__(self, idx):
        return self.input[idx], self.target[idx]

########################################################################################################################


if __name__ == '__main__':
    print('Configs:\n', configs.__dict__)

    # Load England test data (already windowed)
    print('\nLoading England test data...')
    uv_test = np.load("mfwpn_data_england/npy_files/test_wind.npy").astype(np.float32)
    zt_test = np.load("mfwpn_data_england/npy_files/test_pressure.npy").astype(np.float32)

    print(f'Wind shape: {uv_test.shape}')
    print(f'Pressure shape: {zt_test.shape}')

    # Concatenate along channel dimension (axis=2)
    uv_test = np.concatenate((uv_test, zt_test), axis=2)
    print(f'Combined shape: {uv_test.shape}')
    del zt_test

    # Load elevation
    ele = np.load('mfwpn_data_england/npy_files/elevation.npy').astype(np.float32)
    ele[ele < 0] = 0
    ele = (ele - ele.mean()) / ele.std()

    # Data is already windowed, just split into input/target
    print('\nSplitting into input/target...')
    test_x = uv_test[:, :24, :, :, :]  # First 24 hours
    test_y = uv_test[:, 24:, :, :, :]  # Last 24 hours
    del uv_test

    dataset_test = dataset_package(train_x=test_x, train_y=test_y)
    del test_x, test_y
    print('Dataset_test Shape:\n', dataset_test.GetDataShape())

    # Load trained model
    trainer = Trainer(configs)
    checkpoint_path = 'chkfile/checkpoint_mfwpn_england.chk'

    if not os.path.exists(checkpoint_path):
        print(f'\nERROR: Checkpoint not found at {checkpoint_path}')
        print('Available checkpoints:')
        if os.path.exists('chkfile'):
            for f in os.listdir('chkfile'):
                print(f'  - chkfile/{f}')
        exit(1)

    print(f'\nLoading checkpoint from {checkpoint_path}...')
    net = torch.load(checkpoint_path, map_location=configs.device)
    trainer.network.load_state_dict(net['net'])
    print('Model loaded successfully')

    # Run inference
    elev = torch.tensor(ele)
    data = DataLoader(dataset_test, batch_size=configs.batch_size_test, shuffle=False)

    print('\nRunning inference on test set...')
    loss_u_test, loss_v_test = trainer.infer(dataset=dataset_test, dataloader=data, ele=elev)

    loss_test = loss_u_test + loss_v_test
