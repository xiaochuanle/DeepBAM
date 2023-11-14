import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from torch import Tensor
from typing import Any, Callable, List, Optional, Type, Union

import torch.utils
import torch.utils.checkpoint


class ModelBiLSTM(nn.Module):
    def __init__(self, seq_len=51,
                 signal_len=15,
                 num_layers1=3,
                 num_layers2=2,
                 num_classes=2,
                 dropout_rate=0.5,
                 hidden_size=256,
                 vocab_size=16,
                 embedding_size=8,
                 is_base=True,
                 is_signallen=True,
                 is_trace=True,
                 module="both_bilstm",
                 device=0):
        super(ModelBiLSTM, self).__init__()
        self.model_type = 'BiLSTM'
        self.module = module
        self.device = device

        self.seq_len = seq_len
        self.signal_len = signal_len
        self.num_layers1 = num_layers1  # for combined (seq+signal) feature
        self.num_layers2 = num_layers2  # for seq and signal feature separately
        self.num_classes = num_classes

        self.hidden_size = hidden_size

        self.nhid_seq = self.hidden_size // 2
        self.nhid_signal = self.hidden_size - self.nhid_seq

        # seq feature
        if self.module != "signal_bilstm":
            self.embed = nn.Embedding(vocab_size, embedding_size)  # for dna/rna base
            self.is_base = is_base
            self.is_signallen = is_signallen
            self.is_trace = is_trace
            self.sigfea_num = 3 if self.is_signallen else 2
            if self.is_trace:
                self.sigfea_num += 1
            if self.is_base:
                self.lstm_seq = nn.LSTM(embedding_size + self.sigfea_num,
                                        self.nhid_seq,
                                        self.num_layers2,
                                        dropout=dropout_rate,
                                        batch_first=True,
                                        bidirectional=True)
            else:
                self.lstm_seq = nn.LSTM(self.sigfea_num,
                                        self.nhid_seq,
                                        self.num_layers2,
                                        dropout=dropout_rate,
                                        batch_first=True,
                                        bidirectional=True)
            self.fc_seq = nn.Linear(self.nhid_seq * 2, self.nhid_seq)

        # signal feature
        self.lstm_signal = nn.LSTM(self.signal_len,
                                    self.nhid_signal,
                                    self.num_layers2,
                                    dropout=dropout_rate,
                                    batch_first=True,
                                    bidirectional=True)
        self.fc_signal = nn.Linear(self.nhid_signal * 2, self.nhid_signal)

        self.lstm_comb = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layers1,
                                 dropout=dropout_rate, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(hidden_size * 2, num_classes)  # 2 for bidirection

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(1)

    def get_model_type(self):
        return self.model_type

    def forward(self, kmer, 
                signals):
        kmer_embed = self.embed(kmer.long())
        signals = signals.reshape(signals.shape[0], signals.shape[2], signals.shape[3])
        
        out_seq = torch.cat((kmer_embed, signals[:,:, :4]), 2)

        out_signal = signals[:, :, 4:]
        out_seq, _ = self.lstm_seq(out_seq)  # (N, L, nhid_seq*2)
        out_seq = self.fc_seq(out_seq)  # (N, L, nhid_seq)
        out_seq = self.relu(out_seq)

        out_signal, _ = self.lstm_signal(out_signal)
        out_signal = self.fc_signal(out_signal)  # (N, L, nhid_signal)
        out_signal = self.relu(out_signal)

        # combined ================================================
        out = torch.cat((out_seq, out_signal), 2)  # (N, L, hidden_size)
        out, _ = self.lstm_comb(out, )  # (N, L, hidden_size*2)
        out_fwd_last = out[:, -1, :self.hidden_size]
        out_bwd_last = out[:, 0, self.hidden_size:]
        out = torch.cat((out_fwd_last, out_bwd_last), 1)

        # decode
        out = self.dropout1(out)
        out = self.fc1(out)

        return out, self.softmax(out)