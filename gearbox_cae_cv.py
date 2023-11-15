# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:51:18 2023

@author: yunseon

"""

import os
import random
import copy
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import torchsummary

import scipy.signal as signal
from scipy.signal import cwt
from scipy.spatial import distance
from sklearn.metrics import confusion_matrix

# pure_ratio = 0.99; seed_num = 1

results = []
for pure_ratio in [0.99, 0.98, 0.97, 0.96, 0.95]:
    
    for seed_num in [1, 27, 101, 157, 200]:
        
        print(f"ratio : {pure_ratio}", f"seed : {seed_num}")
        
        seed = seed_num
        #파이토치의 랜덤시드 고정
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
        # 넘파이 랜덤시드 고정
        np.random.seed(seed)
        
        # 파이썬 랜덤시드 고정
        random.seed(seed)
        
        ###############################################################################
        
                                        # Data & Params
        
        ###############################################################################
        
        LOAD_PATH = "C:/Users/yunseon/Python/Research/uad_ijpr/gearbox/"
        NUM       = 90
        NUM2      = 90
        WIN       = 256
        
        USE_CUDA = torch.cuda.is_available()
        DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
        EPOCH      = 50
        # BATCH_SIZE = 256
        BATCH_SIZE = 512
        # BATCH_SIZE = 1024
        
        pure     = pure_ratio # normal ratio
        tr_ratio = 0.7        # train data ratio
        N_FEATURE = 4
        
        nm_csv = os.path.join(LOAD_PATH, "Healthy",      f"h30hz{NUM}.csv"); nm_csv = pd.read_csv(nm_csv, encoding = "UTF-8-sig").astype(float)
        ab_csv = os.path.join(LOAD_PATH, "BrokenTooth", f"b30hz{NUM2}.csv"); ab_csv = pd.read_csv(ab_csv, encoding = "UTF-8-sig").astype(float)
        
        ###############################################################
        
        idx = [[i, i + (WIN - 1)] for i in range(0, nm_csv.shape[0] - WIN + 1)]
        
        nm = [nm_csv.loc[i[0] : i[1]].values.T for i in idx]
        nm = np.stack(list(map(lambda x : np.abs(np.fft.fft(x)), nm)))
        
        ###############################################################
        
        idx = [[i, i + (WIN - 1)] for i in range(0, ab_csv.shape[0] - WIN + 1)]
        
        ab = [ab_csv.loc[i[0] : i[1]].values.T for i in idx]
        ab = np.stack(list(map(lambda x : np.abs(np.fft.fft(x)), ab)))
        
        ###############################################################
        
        ########## Define Convolutional Autoencoder
        
        class Convolutional_Autoencoder(nn.Module):
            def __init__(self):
                super(Convolutional_Autoencoder, self).__init__()
        
                self.encoder = nn.Sequential(
                    nn.Conv1d(4, 16, kernel_size = 8, stride = 4, padding = 2),
                    nn.BatchNorm1d(16),
                    nn.LeakyReLU(),
                    nn.Conv1d(16, 8, kernel_size = 8, stride = 4, padding = 2),
                    nn.BatchNorm1d(8),
                    nn.LeakyReLU(),
                    nn.Conv1d(8, 4, kernel_size = 8, stride = 4),
                    nn.BatchNorm1d(4),
                    nn.LeakyReLU(),
        
                )
                
                self.decoder = nn.Sequential(
        
                    nn.ConvTranspose1d(4, 8, kernel_size = 8, stride = 4),
                    nn.BatchNorm1d(8),
                    nn.LeakyReLU(),
                    nn.ConvTranspose1d(8, 16, kernel_size = 8, stride = 4, padding = 2),
                    nn.BatchNorm1d(16),
                    nn.LeakyReLU(),
                    nn.ConvTranspose1d(16, 4, kernel_size = 8, stride = 4, padding = 2),
                )
            
            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return encoded, decoded
        
        
        ###############################################################################
        
                                            # CAE train
        
        ###############################################################################
        
        # train, test dataset
        nm_num = nm.shape[0]
        tr_num = int(nm_num * tr_ratio) 
        
        tr_ab_num = int(tr_num * (1- pure_ratio))
        tr_nm_num = tr_num - tr_ab_num
        
        te_nm_num = nm_num - tr_num
        te_ab_num = te_nm_num
        
        print(f"Train Num : {tr_num}", f"Train Normal : {tr_nm_num}", f"Train Abnormal : {tr_ab_num}")
        print(f"Test Num : {te_nm_num + te_ab_num}",f"Test Normal : {te_nm_num}", f"Test Abnormal : {te_ab_num}")
                
        np.random.shuffle(nm); np.random.shuffle(ab)
        
        nm[:tr_nm_num]
        ab[:tr_ab_num]
        nm[tr_nm_num:tr_nm_num + te_nm_num].shape
        ab[tr_ab_num:tr_ab_num + te_ab_num].shape
        
        train_X = np.vstack([nm[:tr_nm_num], ab[:tr_ab_num]])
        train_Y = np.array(([0] * tr_nm_num) + ([1] * tr_ab_num))
        
        test_X_nm = nm[tr_nm_num : tr_nm_num + te_nm_num]
        test_Y_nm = np.array([0] * len(test_X_nm))
        
        test_X_ab = ab[tr_ab_num:tr_ab_num + te_ab_num]
        test_Y_ab = np.array([1] * len(test_X_nm))
        
        # min_num = min(nm.shape[0], ab.shape[0])
        # tr_num  = int(min_num * tr_ratio)
        # ts_num  = int(min_num * (1 - tr_ratio))
        # print(tr_num, ts_num)
        
        # nm_pure_num = int(tr_num * pure)
        # ab_pure_num = int(tr_num * (1 - pure))
        # print(nm_pure_num, ab_pure_num)
        
        # train_X, train_Y = np.vstack([nm_t[:nm_pure_num], ab_t[:ab_pure_num]]), np.array(([0] * nm_t[:nm_pure_num].shape[0]) + ([1] * ab_t[:ab_pure_num].shape[0]))
        
        # standardize (train)
        mean_vals = np.zeros(train_X.shape[1])
        std_vals = np.zeros(train_X.shape[1])
        
        for i in range(train_X.shape[1]):
            mean_vals[i] = np.mean(train_X[:, i, :])
            std_vals[i] = np.std(train_X[:, i, :])
            train_X[:, i, :] = (train_X[:, i, :] - mean_vals[i])/std_vals[i]
        
        train_X = torch.tensor(train_X, dtype = torch.float32)
        train_Y = torch.tensor(train_Y, dtype = torch.float32)
        print(train_X.shape, train_Y.shape)

        train_dataset = TensorDataset(train_X, train_Y)
        train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, drop_last = False, num_workers = 10, shuffle = True)
        
        # test_X_nm = nm_t[int(min_num - (ts_num/2)):min_num, :, :]
        # test_Y_nm = np.array([0] * nm_t[int(min_num - (ts_num/2)):min_num].shape[0])
        # test_X_ab = ab_t[int(min_num - (ts_num/2)):min_num, :, :]
        # test_Y_ab = np.array([0] * ab_t[int(min_num - (ts_num/2)):min_num].shape[0])
        
        # standardize (test)
        for i in range(train_X.shape[1]):
            test_X_nm[:, i, :] = (test_X_nm[:, i, :] - mean_vals[i])/std_vals[i]
            test_X_ab[:, i, :] = (test_X_ab[:, i, :] - mean_vals[i])/std_vals[i]
        
        test_X_nm = torch.tensor(test_X_nm, dtype = torch.float32)
        test_Y_nm = torch.tensor(test_Y_nm, dtype = torch.float32)
        print(test_X_nm.shape, test_Y_nm.shape)
        
        test_nm_dataset = TensorDataset(test_X_nm, test_Y_nm)
        test_nm_loader  = DataLoader(test_nm_dataset, batch_size = BATCH_SIZE, drop_last = False, num_workers = 10, shuffle = False)
        
        test_X_ab = torch.tensor(test_X_ab, dtype = torch.float32)
        test_Y_ab = torch.tensor(test_Y_ab, dtype = torch.float32)
        print(test_X_ab.shape, test_Y_ab.shape)
        
        test_ab_dataset = TensorDataset(test_X_ab, test_Y_ab)
        test_ab_loader  = DataLoader(test_ab_dataset, batch_size = BATCH_SIZE, drop_last = False, num_workers = 10, shuffle = False)
        
        
        ########## Criterion
        model = Convolutional_Autoencoder().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay = 1e-4)
        criterion = nn.MSELoss()
        
        torchsummary.summary(model, (4, 256))
        
        for epoch in range(1, EPOCH + 1):
            # epoch = 1
            model.train()
            
            e_loss = 0
            for step, (x, y) in enumerate(train_loader):
                
                x = x.to(DEVICE); y = y.to(DEVICE)
        
                encoded, decoded = model(x)
        
                b_loss = criterion(decoded, x) # shape: tensor(num)
                
                optimizer.zero_grad() #기울기에 대한 정보를 초기화합니다.
                b_loss.backward() # 기울기를 구합니다.
                optimizer.step() #최적화를 진행합니다.
                
                e_loss += b_loss
            
            e_loss = e_loss / step
            print(f"{epoch} complete. Loss : {e_loss:.4f}")
        
            if epoch % 10 == 0: 
            
                model.eval()
                train_embed = list(); train_label = list(); train_loss  = list()
                with torch.no_grad():
                    
                    for step, (x, y) in enumerate(train_loader):
                        
                        x = x.to(DEVICE); y = y.to(DEVICE)
                   
                        encoded, decoded = model(x)
                        
                        train_embed.append(encoded.detach().cpu().numpy())
                        train_label.append(y.detach().cpu().numpy())
                        
                        b_loss = torch.abs((x - decoded).detach().cpu() ** 2).reshape(x.shape[0], -1).mean(dim = 1).numpy() # shape: (512,)
                        train_loss.append(b_loss)
                
                        
                train_embed = np.vstack(train_embed)
                train_label = np.concatenate(train_label)
                train_loss = np.concatenate(train_loss)
            
            
                model.eval()
                test_nm_embed = list(); test_nm_loss  = list()
                
                with torch.no_grad():
                    
                    for step, (x, y) in enumerate(test_nm_loader):
                        
                        x = x.to(DEVICE); y = y.to(DEVICE)
                        
                        encoded, decoded = model(x)
                        
                        test_nm_embed.append(encoded.detach().cpu().numpy())
                        # b_loss = torch.abs((y - decoded).detach().cpu()).reshape(BATCH_SIZE, -1).mean(dim = 1).numpy()
                        b_loss = torch.abs((x - decoded).detach().cpu() ** 2).reshape(x.shape[0], -1).mean(dim = 1).numpy() # shape: (512,)
                        test_nm_loss.append(b_loss)
                
                        # plt.plot(x[0].cpu())
                        # plt.plot(decoded[0].cpu())
                        # plt.show()
                
                test_nm_embed = np.vstack(test_nm_embed)
                test_nm_loss = np.concatenate(test_nm_loss)
                
                
                model.eval()
                test_ab_embed = list(); test_ab_loss  = list()
                
                with torch.no_grad():
                    
                    for step, (x, y) in enumerate(test_ab_loader):
                        
                        x = x.to(DEVICE); y = y.to(DEVICE)
                        
                        encoded, decoded = model(x)
                        test_ab_embed.append(encoded.detach().cpu().numpy())
                        # b_loss = torch.abs((y - decoded).detach().cpu()).reshape(BATCH_SIZE, -1).mean(dim = 1).numpy()
                        b_loss = torch.abs((x - decoded).detach().cpu() ** 2).reshape(x.shape[0], -1).mean(dim = 1).numpy() # shape: (512,)
                        test_ab_loss.append(b_loss)
                
                        # plt.plot(x[0].cpu())
                        # plt.plot(decoded[0].cpu())
                        # plt.show()
                
                test_ab_embed = np.vstack(test_ab_embed)
                test_ab_loss = np.concatenate(test_ab_loss)
                
                
                # loss histogram
                plt.title(epoch)
                plt.hist(train_loss, np.arange(0, 1.1, 0.05), alpha = 0.3, label = "tr_loss")
                plt.hist(test_nm_loss, np.arange(0, 1.1, 0.05), alpha = 0.3, label = "ts_nm_loss")
                plt.hist(test_ab_loss, np.arange(0, 1.1, 0.05), alpha = 0.3, label = "ts_ab_loss")
                plt.legend()
                plt.show()
            
                th = np.percentile(train_loss, 95)
                
                real = [0] * len(test_nm_loss)
                pred = list(test_nm_loss > th); pred = [int(i) for i in pred]
                
                real += [1] * len(test_ab_loss)
                pred += list(test_ab_loss > th); pred = [int(i) for i in pred]
                
                tab = pd.crosstab(real, pred)
                
                acc = (tab.iloc[0,0] + tab.iloc[1,1]) / tab.sum().sum()
                precision = tab.iloc[1,1] / (tab.iloc[0,1] + tab.iloc[1,1])
                recall = tab.iloc[1,1] / (tab.iloc[1,0] + tab.iloc[1,1])
                f1 = (2 * precision * recall)/(precision + recall)
            
                print("-----------------------------")
                print(f"EPOCH : {epoch}")
                print(tab, "\n")
                print("acc: ", acc)
                print("precision: ", precision)
                print("recall: ", recall)
                print("f1 score: ", f1)
                print("-----------------------------")
        
        results.append([NUM, NUM2, int((1-pure)*100), seed, acc, precision, recall, f1])

SAVE_PATH = "C:/Users/yunseon/Python/Research/uad_ijpr/result/"
pd.DataFrame(results, columns = ["nm_load", "ab_load", "pure_ratio", "seed", "acc", "precision", "recall", "f1score"]).to_csv(SAVE_PATH + 'cae_k8_s4_b512_5cv.csv')

tmp = pd.DataFrame(results, columns = ["nm_load", "ab_load", "pure_ratio", "seed", "acc", "precision", "recall", "f1score"])
tmp.groupby(["pure_ratio"]).mean()
# tmp.groupby(["pure_ratio"]).std()