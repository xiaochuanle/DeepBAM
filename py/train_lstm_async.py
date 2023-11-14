import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary
from sklearn import metrics
import torchmetrics
import numpy as np
import argparse
import os
import sys
import time
import re
from tqdm import tqdm
from queue import Queue
from threading import Thread
import random
from model import *

from dataloader import *

data_Q = Queue(1)

def load_data_thread(train_dir : str,
                     train_batch_size : int ,
                     valid_batch_size : int,
                     kmer_size : int,  
                     max_epoch : int):
    train_files = [train_dir + x for x in os.listdir(train_dir)]
    for epoch in range(max_epoch):
        random.shuffle(train_files)
        split = int(len(train_files) * 0.7)
        train_list = train_files[: split]
        valid_list = train_files[split :]
        for train_file in train_list:
            train_dataset = Dataset_npy(train_file, kmer = kmer_size)
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                       batch_size = train_batch_size,
                                                       num_workers = 16,
                                                       pin_memory = True,
                                                       shuffle = True)
            data_Q.put((train_loader, "train"))
        for valid_file in valid_list:
            valid_dataset = Dataset_npy(valid_file, kmer=kmer_size)
            valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                        batch_size=valid_batch_size,
                                                        num_workers = 16, 
                                                        pin_memory=True, 
                                                        shuffle=True)
            data_Q.put((valid_loader, "valid"))
        data_Q.put((None, "next"))




def train_data_thread(model,
                      model_save : str ,
                      model_type : str , 
                      signal_len : int ,
                      kmer_size : int ,
                      max_epoch = 25,  
                      ):
    f_accuracy = torchmetrics.Accuracy('binary', num_classes=2).cuda()
    f_precision = torchmetrics.Precision("binary", num_classes=2).cuda()
    f_recall = torchmetrics.Recall('binary', num_classes=2).cuda()
    f_F1_score = torchmetrics.F1Score('binary', num_classes=2).cuda()
    model = model.cuda()
    weight_rank = torch.from_numpy(np.array([1, 1])).float()
    weight_rank = weight_rank.cuda()
    criterion = nn.CrossEntropyLoss(weight=weight_rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.2)
    global_best_accuracy = 0
    curr_epoch_accuracy = 0
    model.train()
    for epoch in range(max_epoch):
        tlosses = torch.tensor([]).cuda()
        vlosses, vaccus, vprecs, vrecas, vf1score = torch.tensor([]).cuda(), \
            torch.tensor([]).cuda(), torch.tensor([]).cuda(), \
                torch.tensor([]).cuda(), torch.tensor([]).cuda()
        start = time.time()
        data_loader, flag = data_Q.get()
        while flag != "next":
            # LOGGER.info("Start to process data")
            if flag == "train":
                model.train()
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
            elif flag == "valid":
                model.eval()
                with torch.no_grad():
                    for vi, vsfeatures in enumerate(data_loader):
                        vkmer, vsignals, vlabels = vsfeatures
                        vkmer = vkmer.cuda()
                        vsignals = vsignals.float().cuda()
                        vlabels = vlabels.long().cuda()
                        voutputs, vlogits = model(vkmer, vsignals)
                        
                        vloss = criterion(voutputs, vlabels)

                        _, vpredicted = torch.max(vlogits.data, 1)
                        vpredicted = vpredicted.cuda()

                        vaccus = torch.concatenate((vaccus, f_accuracy(vpredicted, vlabels).reshape(1)))
                        vprecs = torch.concatenate((vprecs, f_precision(vpredicted, vlabels).reshape(1)))
                        vrecas = torch.concatenate((vrecas, f_recall(vpredicted, vlabels).reshape(1)))
                        vf1score = torch.concatenate((vf1score, f_F1_score(vpredicted, vlabels).reshape(1)))
                        vlosses = torch.concatenate((vlosses, vloss.detach().reshape(1)))
            # LOGGER.info("process data done!")
            data_loader, flag = data_Q.get()
        curr_epoch_accuracy = torch.mean(vaccus)
        if curr_epoch_accuracy > global_best_accuracy - 0.0002:
            torch.save(model.state_dict(),
                        model_save + model_type + '_b{}_s{}_epoch{}_accuracy:{:.4f}.pt'.format(kmer_size,
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
        f_F1_score.reset()
        start = time.time()
        sys.stdout.flush()
        scheduler.step()


if __name__ == "__main__":
    """
    this is a python script for train lstm model from scratch, 
    the data is split into separate chunks due to its large memory usage
    """
    train_dir = "/mnt/sde2/SugarbeetCG/merge_hg002_and_sugarbeet/"
    save_dir = "/public2/YHC/all_model_2/"
    model_type = "LSTM"
    signal_len = 15
    kmer_size = 51
    train_batch_size = 1024
    valid_batch_size = 2048
    max_epoch = 25
    model = ModelBiLSTM()

    load_thread = Thread(target=load_data_thread,args=(train_dir,
                                                train_batch_size,
                                                valid_batch_size,
                                                kmer_size,
                                                max_epoch))
    train_thread = Thread(target=train_data_thread, args=(model,
                                                   save_dir,
                                                   model_type,
                                                   signal_len,
                                                   kmer_size,
                                                   max_epoch))
    
    train_thread.start()
    load_thread.start()
    
    train_thread.join()
    load_thread.join()




