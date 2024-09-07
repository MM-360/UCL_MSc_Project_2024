import os
import dill
import sys
import pickle

import torch
import torch.optim as optim
import torch.nn as nn

from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader,TensorDataset
from torchvision import datasets,transforms
from torchvision.transforms import v2
from torchsummary import summary

class lstm_latent_2(nn.Module):
    """
    3 LSTM Cells

    2 Fully connected layers

    Showed best results in testings
    
    """
    def __init__(self, n_hidden = 256, dim_latent =64):

        super().__init__()
        self.n_hidden = n_hidden

        self.lstm1 = nn.LSTMCell(dim_latent, int(self.n_hidden*0.5))
        self.lstm2 = nn.LSTMCell(int(self.n_hidden*0.5), self.n_hidden)
        self.lstm3 = nn.LSTMCell(self.n_hidden, self.n_hidden*2)
        #self.lstm3 = nn.LSTMCell(self.n_hidden*2, self.n_hidden*2)
        self.linear1 = nn.Linear(self.n_hidden*2,self.n_hidden)
        self.linear2 = nn.Linear(self.n_hidden,dim_latent)

    def forward(self, X, future = 0, device = 'cpu'):
        outputs = []
        n_samples = X.shape[0]

        h_t = torch.zeros(n_samples,int(self.n_hidden*0.5), dtype=torch.float32).to(device)
        c_t = torch.zeros(n_samples, int(self.n_hidden*0.5), dtype=torch.float32).to(device)
        h_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32).to(device)
        c_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32).to(device)
        h_t3 = torch.zeros(n_samples, self.n_hidden*2, dtype=torch.float32).to(device)
        c_t3 = torch.zeros(n_samples, self.n_hidden*2, dtype=torch.float32).to(device)


        for input_t in X.split(1,dim = 1 ):
            h_t, c_t = self.lstm1(input_t.squeeze(dim = 1), (h_t,c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2,c_t2))
            h_t3, c_t3 = self.lstm3(h_t2, (h_t3,c_t3))
            #h_t4, c_t4 = self.lstm3(h_t2, (h_t3,c_t3))

            output = self.linear1(h_t3)
            output = self.linear2(output)
            outputs.append(output)
        #h1 = self.layers2(h1)
        for i in range(future):
            h_t, c_t = self.lstm1(output, (h_t,c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2,c_t2))
            h_t3, c_t3 = self.lstm3(h_t2, (h_t3,c_t3))
            output = self.linear1(h_t3)
            output = self.linear2(output)
            outputs.append(output)

        return torch.stack(outputs)



class lstm_latent(nn.Module):
    """
    6 LSTM Cells

    2 Fully connected layers

    Showed ok results in testings
    
    """
    def __init__(self, n_hidden = 256, dim_latent = 64):
    
        super().__init__()
        self.n_hidden = n_hidden
        self.dim_latent = dim_latent

        self.lstm1 = nn.LSTMCell(self.dim_latent, int(self.n_hidden*0.5))
        self.lstm2 = nn.LSTMCell(int(self.n_hidden*0.5), self.n_hidden)
        self.lstm3 = nn.LSTMCell(self.n_hidden, self.n_hidden*2)
        self.lstm4 = nn.LSTMCell(self.n_hidden*2, self.n_hidden*2)
        self.lstm5 = nn.LSTMCell(self.n_hidden*2, self.n_hidden*2)
        self.lstm6 = nn.LSTMCell(self.n_hidden*2, self.n_hidden)
        self.linear1 = nn.Linear(self.n_hidden,int(self.n_hidden*0.5))
        self.linear2 = nn.Linear(int(self.n_hidden*0.5),        self.dim_latent)
        #self.linear3 = nn.Linear(self.n_hidden,64)

    def forward(self, X, future = 0, device = 'cpu'):
        outputs = []
        n_samples = X.shape[0]

        h_t = torch.zeros(n_samples,int(self.n_hidden*0.5), dtype=torch.float32).to(device)
        c_t = torch.zeros(n_samples, int(self.n_hidden*0.5), dtype=torch.float32).to(device)
        h_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32).to(device)
        c_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32).to(device)
        h_t3 = torch.zeros(n_samples, self.n_hidden*2, dtype=torch.float32).to(device)
        c_t3 = torch.zeros(n_samples, self.n_hidden*2, dtype=torch.float32).to(device)
        h_t4 = torch.zeros(n_samples, self.n_hidden*2, dtype=torch.float32).to(device)
        c_t4 = torch.zeros(n_samples, self.n_hidden*2, dtype=torch.float32).to(device)
        h_t5 = torch.zeros(n_samples, self.n_hidden*2, dtype=torch.float32).to(device)
        c_t5 = torch.zeros(n_samples, self.n_hidden*2, dtype=torch.float32).to(device)
        h_t6 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32).to(device)
        c_t6 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32).to(device)

        for input_t in X.split(1,dim = 1 ):
            h_t, c_t = self.lstm1(input_t.squeeze(dim = 1), (h_t,c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2,c_t2))
            h_t3, c_t3 = self.lstm3(h_t2, (h_t3,c_t3))
            h_t4, c_t4 = self.lstm4(h_t3, (h_t4,c_t4))
            h_t5, c_t5 = self.lstm5(h_t4, (h_t5,c_t5))
            h_t6, c_t6 = self.lstm6(h_t5, (h_t6,c_t6))

            output = self.linear1(h_t6)
            output = self.linear2(output)
            outputs.append(output)

        for i in range(future):
            h_t, c_t = self.lstm1(output, (h_t,c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2,c_t2))
            h_t3, c_t3 = self.lstm3(h_t2, (h_t3,c_t3))
            h_t4, c_t4 = self.lstm4(h_t3, (h_t4,c_t4))
            h_t5, c_t5 = self.lstm5(h_t4, (h_t5,c_t5))
            h_t6, c_t6 = self.lstm6(h_t5, (h_t6,c_t6))
            output = self.linear1(h_t6)
            output = self.linear2(output)
            outputs.append(output)
        
        return torch.stack(outputs)


class lstm_latent_3(nn.Module):
    """
    3 LSTM Cells

    1 Fully connected layers
    
    """
    def __init__(self, n_hidden = 256, dim_latent =64):

        super().__init__()
        self.n_hidden = n_hidden
        self.dim_latent = dim_latent
        

        self.lstm1 = nn.LSTMCell(dim_latent, int(self.n_hidden*0.5))
        self.lstm2 = nn.LSTMCell(int(self.n_hidden*0.5), int(self.n_hidden*0.5))
        self.lstm3 = nn.LSTMCell(int(self.n_hidden*0.5), self.n_hidden)
        self.linear1 = nn.Linear(self.n_hidden,self.dim_latent )

    def forward(self, X, future = 0, device = 'cpu'):
        outputs = []
        n_samples = X.shape[0]

        h_t = torch.zeros(n_samples,int(self.n_hidden*0.5), dtype=torch.float32).to(device)
        c_t = torch.zeros(n_samples, int(self.n_hidden*0.5), dtype=torch.float32).to(device)
        h_t2 = torch.zeros(n_samples, int(self.n_hidden*0.5), dtype=torch.float32).to(device)
        c_t2 = torch.zeros(n_samples, int(self.n_hidden*0.5), dtype=torch.float32).to(device)
        h_t3 = torch.zeros(n_samples, self.n_hidden*1, dtype=torch.float32).to(device)
        c_t3 = torch.zeros(n_samples, self.n_hidden*1, dtype=torch.float32).to(device)


        for input_t in X.split(1,dim = 1 ):
            h_t, c_t = self.lstm1(input_t.squeeze(dim = 1), (h_t,c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2,c_t2))
            h_t3, c_t3 = self.lstm3(h_t2, (h_t3,c_t3))

            output = self.linear1(h_t3)
            outputs.append(output)

        for i in range(future):
            h_t, c_t = self.lstm1(output, (h_t,c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2,c_t2))
            h_t3, c_t3 = self.lstm3(h_t2, (h_t3,c_t3))
            output = self.linear1(h_t3)
            outputs.append(output)

        return torch.stack(outputs)


class lstm_latent_4(nn.Module):
    """
    3 LSTM Cells

    1 Fully connected layers
    
    """
    def __init__(self, n_hidden = 256, dim_latent =64):

        super().__init__()
        self.n_hidden = n_hidden
        self.dim_latent = dim_latent
        

        self.lstm1 = nn.LSTMCell(dim_latent, int(self.n_hidden*0.5))
        self.lstm2 = nn.LSTMCell(int(self.n_hidden*0.5), int(self.n_hidden*1))
        self.lstm3 = nn.LSTMCell(int(self.n_hidden*1), self.n_hidden*2)
        self.linear1 = nn.Linear(int(self.n_hidden*2),self.dim_latent )

    def forward(self, X, future = 0, device = 'cpu'):
        outputs = []
        n_samples = X.shape[0]

        h_t = torch.zeros(n_samples,int(self.n_hidden*0.5), dtype=torch.float32).to(device)
        c_t = torch.zeros(n_samples, int(self.n_hidden*0.5), dtype=torch.float32).to(device)
        h_t2 = torch.zeros(n_samples, int(self.n_hidden*1), dtype=torch.float32).to(device)
        c_t2 = torch.zeros(n_samples, int(self.n_hidden*1), dtype=torch.float32).to(device)
        h_t3 = torch.zeros(n_samples, self.n_hidden*2, dtype=torch.float32).to(device)
        c_t3 = torch.zeros(n_samples, self.n_hidden*2, dtype=torch.float32).to(device)


        for input_t in X.split(1,dim = 1 ):
            h_t, c_t = self.lstm1(input_t.squeeze(dim = 1), (h_t,c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2,c_t2))
            h_t3, c_t3 = self.lstm3(h_t2, (h_t3,c_t3))

            output = self.linear1(h_t3)
            outputs.append(output)

        for i in range(future):
            h_t, c_t = self.lstm1(output, (h_t,c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2,c_t2))
            h_t3, c_t3 = self.lstm3(h_t2, (h_t3,c_t3))
            output = self.linear1(h_t3)
            outputs.append(output)

        return torch.stack(outputs)