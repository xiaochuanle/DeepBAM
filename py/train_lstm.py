import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary
import torchmetrics
import numpy as np
import argparse
import os
import sys
import time
import re
from tqdm import tqdm
import random

from model import *
from dataloader import *

# input your own data path here
train_dir_1 = ""
train_dir_2 = ""

kmer_size = 51

model_save = "/public2/YHC/all_model_7/"
model_type = "LSTM_20240119_"

signal_len = 15
f_accuracy = torchmetrics.Accuracy('binary', num_classes=2).cuda()
f_precision = torchmetrics.Precision("binary", num_classes=2).cuda()
f_recall = torchmetrics.Recall('binary', num_classes=2).cuda()
f_F1_score = torchmetrics.F1Score('binary', num_classes=2).cuda()

train_batch_size = 1024
valid_batch_size = 1024
max_epoch = 25
SEED = 71
model = ModelBiLSTM()
model = model.to(torch.device("cuda"))

train_files_1 = [train_dir_1 + x for x in os.listdir(train_dir_1) if x.endswith("npy")]
train_files_2 = [train_dir_2 + x for x in os.listdir(train_dir_2) if x.endswith("npy")]

random.seed(71)

step_interval = 4
counter = 0
step = 0
weight_rank = torch.from_numpy(np.array([1, 1])).float()
weight_rank = weight_rank.cuda()
criterion = nn.CrossEntropyLoss(weight=weight_rank)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=1, gamma=0.4)
curr_best_accuracy = 0
model.train()
epoch = 0
global_best_accuracy = 0
for epoch in range(max_epoch):
    
    random.shuffle(train_files_1)
    random.shuffle(train_files_2)
    train_files = []
    for i in range(len(train_files_2)):
        train_files.append([train_files_1[i], train_files_2[i]])
    curr_epoch_accuracy = 0
    no_best_model = True
    tlosses = torch.tensor([]).cuda()
    start = time.time()
    # k-fold training
    random.shuffle(train_files)
    split = int(0.7 * len(train_files))
    train_list = train_files[:split]
    valid_list = train_files[split:]
    for train_file in train_list:
        print("Reading k-fold train_data from: {} and {}".format(train_file[0], train_file[1]))
        dataset_ = Dataset_npy(train_file[0], train_file[1], kmer=kmer_size)
        data_loader = torch.utils.data.DataLoader(dataset = dataset_,
                                                batch_size = train_batch_size,
                                                num_workers = 16, 
                                                pin_memory = True,
                                                shuffle = True)
        for i, sfeatures in enumerate(data_loader):
            kmer, signals, labels = sfeatures
            kmer = kmer.cuda()
            signals = signals.float().cuda()
            labels = labels.long().cuda()
            outputs, _ = model(kmer, signals)
            loss = criterion(outputs, labels)
            tlosses = torch.concatenate((tlosses, loss.detach().reshape(1)))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), .7)
            optimizer.step()
                
    model.eval()
    with torch.no_grad():
        vlosses, vaccus, vprecs, vrecas, vf1score = torch.tensor([]).cuda(), \
            torch.tensor([]).cuda(), torch.tensor([]).cuda(), \
                torch.tensor([]).cuda(), torch.tensor([]).cuda()
        for valid_file in valid_list:
            print("Read k-fold valid_data from-{} and {}".format(valid_file[0], valid_file[1]))
            dataset_ = Dataset_npy(valid_file[0], valid_file[1], kmer=kmer_size)
            data_loader = torch.utils.data.DataLoader(dataset = dataset_,
                                                        batch_size = valid_batch_size,
                                                        num_workers = 16, 
                                                        pin_memory = True, 
                                                        shuffle = True)
            for vi, vsfeatures in enumerate(data_loader):
                kmer, signals, vlabels = vsfeatures
                kmer = kmer.cuda()
                signals = signals.float().cuda()
                vlabels = vlabels.long().cuda()                
                voutputs, vlogits = model(kmer, signals)
                vloss = criterion(voutputs, vlabels)

                _, vpredicted = torch.max(vlogits.data, 1)
                vpredicted = vpredicted.cuda()

                vaccus = torch.concatenate((vaccus, f_accuracy(vpredicted, vlabels).reshape(1)))
                vprecs = torch.concatenate((vprecs, f_precision(vpredicted, vlabels).reshape(1)))
                vrecas = torch.concatenate((vrecas, f_recall(vpredicted, vlabels).reshape(1)))
                vf1score = torch.concatenate((vf1score, f_F1_score(vpredicted, vlabels).reshape(1)))
                vlosses = torch.concatenate((vlosses, vloss.detach().reshape(1)))

        curr_epoch_accuracy = torch.mean(vaccus)
        if curr_epoch_accuracy > global_best_accuracy - 0.0002:
            traced_script_module = torch.jit.trace(model, (kmer, signals))
            torch.save(model.state_dict(),
                        model_save + model_type + '_b{}_s{}_epoch{}_accuracy:{:.4f}.pt'.format(kmer_size,
                                                signal_len,
                                                epoch + 1, curr_epoch_accuracy))
            traced_script_module.save(model_save + model_type + 'script_b{}_s{}_epoch{}_accuracy:{:.4f}.pt'.format(kmer_size,
                                                signal_len,
                                                epoch + 1, curr_epoch_accuracy))
            global_best_accuracy = curr_epoch_accuracy
            
    time_cost = time.time() - start
    print('Epoch [{}/{}], TrainLoss: {:.4f}; '
            'ValidLoss: {:.4f}, '
            'Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1-score: {:.4f}, '
            'curr_epoch_best_accuracy: {:.4f}; Time: {:.2f}s'
            .format(epoch + 1, max_epoch, torch.mean(tlosses),
                    torch.mean(vlosses), torch.mean(vaccus), torch.mean(vprecs), 
                    torch.mean(vrecas), torch.mean(vf1score), curr_epoch_accuracy, time_cost))
    f_accuracy.reset()
    f_precision.reset()
    f_recall.reset()
    tlosses = torch.tensor([]).cuda()
    start = time.time()
    sys.stdout.flush()
    scheduler.step()
    model.train()
