import os
import dill
import sys
import pickle

import torch
import torch.optim as optim
import torch.nn as nn

from torch.utils.data import DataLoader,TensorDataset

class Encoder(nn.Module):
    
    def __init__(self, d_latent = 64):
        """ 
        Encoder using `nn.Module`. Series of 2d-Convolutions, dropouts and Relus

        """
        super().__init__()
        self.d_latent = d_latent
        self.layers = nn.Sequential(
                                    nn.Conv2d(1, 8, kernel_size=5),
                                    nn.Conv2d(8, 8, kernel_size=5),
                                    nn.Conv2d(8, 8, kernel_size=5),
                                    nn.BatchNorm2d(8),
                                    nn.ReLU(),
                                    nn.Conv2d(8, 16, kernel_size=5, stride=1),
                                    nn.Conv2d(16, 16, kernel_size=5, stride=1),
                                    nn.Conv2d(16, 16, kernel_size=5, stride=1),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU(),
                                    nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=2),
                                    nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=2),
                                    nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=2),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.Conv2d(32, 64, kernel_size=3),
                                    nn.Conv2d(64, 64, kernel_size=3),
                                    nn.Conv2d(64, 64, kernel_size=3),
                                    nn.BatchNorm2d(64),
                                    nn.Flatten(),
                                    nn.Linear(2304, d_latent),
                                    )
        self.sigmoid=nn.Sigmoid()
    def forward(self, X):
        h1 = self.layers(X)
        #h1 = self.layers2(h1)
        h1 =self.sigmoid(h1)
        return h1
    

class Decoder(nn.Module):
   
    def __init__(self,d_latent = 64):
        """ 

        """
        super().__init__()
        self.d_latent = d_latent
        self.layers = nn.Sequential(
                                    nn.Linear(self.d_latent, 2304),
                                    nn.ReLU(),
                                    nn.Unflatten(1, (64, 6, 6)),
                                    nn.ConvTranspose2d(64, 16, kernel_size=3), #stride = 3),
                                    nn.ConvTranspose2d(16, 16,kernel_size=3), #stride = 3),
                                    nn.ConvTranspose2d(16, 16,kernel_size=3, stride = 3),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(16, 8, kernel_size=3),#, stride = 2),
                                    nn.ConvTranspose2d(8, 8, kernel_size=3),#, stride = 2),
                                    nn.ConvTranspose2d(8, 8, kernel_size=3, stride = 3),
                                    nn.BatchNorm2d(8),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(8, 4, kernel_size=3),
                                    nn.ConvTranspose2d(4, 4, kernel_size=3),
                                    nn.BatchNorm2d(4),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(4, 1, kernel_size=2,padding =1),#,stride = 3),
                                    #nn.ConvTranspose2d(4, 1, kernel_size=3),#,padding =19),
                                    nn.BatchNorm2d(1)
                                    )
    def forward(self, Z):
        h1 = self.layers(Z)
        #h1 = self.sigmoid(h1)
        return h1
   

class Decoder_sigmoid(nn.Module):
   
    def __init__(self,d_latent = 64):
        """ 

        """
        super().__init__()
        self.d_latent = d_latent
        self.layers = nn.Sequential(
                                    nn.Linear(self.d_latent, 2304),
                                    nn.ReLU(),
                                    nn.Unflatten(1, (64, 6, 6)),
                                    nn.ConvTranspose2d(64, 16, kernel_size=3), #stride = 3),
                                    nn.ConvTranspose2d(16, 16,kernel_size=3), #stride = 3),
                                    nn.ConvTranspose2d(16, 16,kernel_size=3, stride = 3),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(16, 8, kernel_size=3),#, stride = 2),
                                    nn.ConvTranspose2d(8, 8, kernel_size=3),#, stride = 2),
                                    nn.ConvTranspose2d(8, 8, kernel_size=3, stride = 3),
                                    nn.BatchNorm2d(8),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(8, 4, kernel_size=3),
                                    nn.ConvTranspose2d(4, 4, kernel_size=3),
                                    nn.BatchNorm2d(4),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(4, 1, kernel_size=2,padding =1),#,stride = 3),
                                    #nn.ConvTranspose2d(4, 1, kernel_size=3),#,padding =19),
                                    nn.BatchNorm2d(1)
                                    )
        self.sigmoid=nn.Sigmoid()
    def forward(self, Z):
        h1 = self.layers(Z)
        h1 = self.sigmoid(h1)
        return h1

class MLP_enc_SIC(nn.Module):
    
    def __init__(self,d_latent = 64, img_size = 105):
        super().__init__()
        self.d_latent = d_latent
        #self.linear =   nn.Linear()
        #self.relu =  nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.linear1 = nn.Sequential(nn.Flatten(), nn.Linear(in_features = img_size*img_size, out_features = 1024))
        self.linear2 = nn.Sequential(nn.ReLU(),nn.Linear(in_features = 1024, out_features = 512))
        self.linear3 = nn.Sequential(nn.ReLU(),nn.Linear(in_features = 512, out_features = self.d_latent))

    def forward(self, Z):
        h1 = self.linear1(Z)
        h1 = self.linear2(h1)
        h1 = self.linear3(h1)
        h1 = self.sigmoid(h1)
        return h1

class MLP_dec_SIC(nn.Module):
    
    def __init__(self,d_latent = 64, img_size = 105):
        super().__init__()
        self.d_latent = d_latent
        self.linear1 = nn.Sequential(nn.Linear(in_features = self.d_latent, out_features = 512) , nn.ReLU())
        self.linear2 = nn.Sequential(nn.Linear(in_features = 512 , out_features = 1024), nn.ReLU())
        self.linear3 = nn.Sequential(nn.Linear(in_features =1024, out_features = img_size*img_size), nn.Flatten())

    def forward(self, Z, img_size = 105):
        h1 = self.linear1(Z)
        h1 = self.linear2(h1)
        h1 = self.linear3(h1)
        return h1.reshape(-1,1,img_size,img_size)