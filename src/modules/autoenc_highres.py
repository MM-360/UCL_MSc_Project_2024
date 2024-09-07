import os
import dill
import sys
import pickle

import torch
import torch.optim as optim
import torch.nn as nn

from torch.utils.data import DataLoader,TensorDataset

class Encoder_sharp(nn.Module):
    
    def __init__(self, d_latent = 64):
        """ 
        Encoder using `nn.Module`. Series of 2d-Convolutions, dropouts and Relus

        """
        super().__init__()
        self.d_latent = d_latent
        self.layers = nn.Sequential(
                                        # nn.Conv2d(1, 8, kernel_size=3),
                                        # nn.ReLU(),
                                        #nn.MaxPool2d(2),
                                        nn.Conv2d(1, 16, kernel_size=3),
                                        nn.ReLU(),
                                        nn.MaxPool2d(2),
                                        nn.BatchNorm2d(16),
                                        nn.Conv2d(16, 32, kernel_size=3),
                                        nn.ReLU(),
                                        nn.MaxPool2d(2),
                                        nn.Conv2d(32, 64, kernel_size=3),
                                        nn.ReLU(),
                                        nn.MaxPool2d(2),
                                        nn.BatchNorm2d(64),
                                        # nn.Conv2d(64, 64, kernel_size=3),
                                        # nn.ReLU(),
                                        # nn.MaxPool2d(2),
                                        # nn.BatchNorm2d(64),
                                        nn.Flatten(),
                                        nn.Linear(87616, self.d_latent)
                                    )
        self.sigmoid=nn.Sigmoid()
    def forward(self, X):
        h1 = self.layers(X)
        h1 =self.sigmoid(h1)
        return h1


class Decoder_sharp(nn.Module):
    # no max pool used in encoder
    def __init__(self,d_latent = 64):
        """ 

        """
        super().__init__()
        self.d_latent = d_latent
        self.layers = nn.Sequential(
                                    nn.Linear(self.d_latent, 87616),
                                    nn.ReLU(),
                                    nn.Unflatten(1, (64, 37, 37)),
                                    nn.ConvTranspose2d(64, 32, kernel_size=5, stride = 2),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(32, 16, kernel_size=5, stride = 2),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(16, 8, kernel_size=3, stride = 2),
                                    nn.BatchNorm2d(8),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(8, 4, kernel_size=3),
                                    nn.BatchNorm2d(4),
                                    nn.ReLU(),
                                    nn.Conv2d(4, 1, kernel_size=3),
                                    nn.Conv2d(1, 1, kernel_size=2),
                                    nn.BatchNorm2d(1)
                                    )    
    def forward(self, Z):
        h1 = self.layers(Z)
        return h1
    
class Encoder_sharp_1024(nn.Module):
    
    def __init__(self, d_latent = 64):
        """ 
        Encoder using `nn.Module`. Series of 2d-Convolutions, dropouts and Relus

        """
        super().__init__()
        self.d_latent = d_latent
        self.layers = nn.Sequential(
                                    # nn.Conv2d(1, 8, kernel_size=3),
                                    # nn.ReLU(),
                                    #nn.MaxPool2d(2),
                                    nn.Conv2d(1, 8, kernel_size=5),
                                    nn.Conv2d(8, 8, kernel_size=5),
                                    nn.Conv2d(8, 8, kernel_size=5),
                                    #nn.MaxPool2d(2),
                                    nn.BatchNorm2d(8),
                                    nn.ReLU(),
                                    nn.Conv2d(8, 16, kernel_size=5, stride=1),
                                    nn.Conv2d(16, 16, kernel_size=5, stride=1),
                                    nn.Conv2d(16, 16, kernel_size=5, stride=1),
                                    nn.MaxPool2d(2),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU(),
                                    nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=2),
                                    nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=2),
                                    nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=2),
                                    nn.MaxPool2d(2),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.Conv2d(32, 64, kernel_size=3),
                                    nn.Conv2d(64, 64, kernel_size=3),
                                    nn.Conv2d(64, 64, kernel_size=3),
                                    #nn.MaxPool2d(2),
                                    nn.BatchNorm2d(64),
                                    nn.Flatten(),
                                    nn.Linear(1024, d_latent),
                                    )
        self.sigmoid=nn.Sigmoid()

        # self.layers_dec = nn.Sequential(
        #                     nn.Linear(self.d_latent, 1024),
        #                     nn.ReLU(),
        #                     nn.Unflatten(1, (64, 4, 4)),
        #                     nn.ConvTranspose2d(64, 32, kernel_size=3), #stride = 3),
        #                     nn.ConvTranspose2d(32, 32, kernel_size=3),#, stride = 2),
        #                     nn.ConvTranspose2d(32, 32, kernel_size=3, stride = 4),
        #                     nn.BatchNorm2d(32),
        #                     nn.ReLU(),
        #                     nn.ConvTranspose2d(32, 16,kernel_size=3), #stride = 3),
        #                     nn.ConvTranspose2d(16, 16,kernel_size=3),
        #                     nn.ConvTranspose2d(16, 16,kernel_size=3, stride = 4),
        #                     nn.BatchNorm2d(16),
        #                     nn.ReLU(),
        #                     nn.ConvTranspose2d(16, 8, kernel_size=5),#, stride = 2),
        #                     nn.ConvTranspose2d(8, 8, kernel_size=5),#, stride = 2),
        #                     nn.ConvTranspose2d(8, 8, kernel_size=5, stride = 2),
        #                     nn.BatchNorm2d(8),
        #                     nn.ReLU(),
        #                     nn.ConvTranspose2d(8, 4, kernel_size=5),
        #                     nn.ConvTranspose2d(4, 4, kernel_size=5),
        #                     nn.ConvTranspose2d(4, 4, kernel_size=5),
        #                     nn.ConvTranspose2d(4, 1, kernel_size=5), #padding = 1),
        #                     nn.ConvTranspose2d(1, 1, kernel_size=2), #padding = 1),
        #                     nn.BatchNorm2d(1)
        #                     )


    def forward(self, X):
        h1 = self.layers(X)
        h1 = self.sigmoid(h1)
        #h1 = self.layers_dec(h1)
        return h1
    

class Decoder_sharp_1024(nn.Module):
    # no max pool used in encoder
    def __init__(self,d_latent = 64):
        """ 

        """
        super().__init__()
        self.d_latent = d_latent
        self.layers = nn.Sequential(
                                    nn.Linear(self.d_latent, 1024),
                                    nn.ReLU(),
                                    nn.Unflatten(1, (64, 4, 4)),
                                    nn.ConvTranspose2d(64, 32, kernel_size=3), #stride = 3),
                                    nn.ConvTranspose2d(32, 32, kernel_size=3),#, stride = 2),
                                    nn.ConvTranspose2d(32, 32, kernel_size=3, stride = 4),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(32, 16,kernel_size=3), #stride = 3),
                                    nn.ConvTranspose2d(16, 16,kernel_size=3),
                                    nn.ConvTranspose2d(16, 16,kernel_size=3, stride = 4),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(16, 8, kernel_size=5),#, stride = 2),
                                    nn.ConvTranspose2d(8, 8, kernel_size=5),#, stride = 2),
                                    nn.ConvTranspose2d(8, 8, kernel_size=5, stride = 2),
                                    nn.BatchNorm2d(8),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(8, 4, kernel_size=5),
                                    nn.ConvTranspose2d(4, 4, kernel_size=5),
                                    nn.ConvTranspose2d(4, 4, kernel_size=5),
                                    nn.BatchNorm2d(4),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(4, 1, kernel_size=5), #padding = 1),
                                    nn.ConvTranspose2d(1, 1, kernel_size=2), #padding = 1),
                                    #nn.BatchNorm2d(1)
                                    )
    def forward(self, Z):
        h1 = self.layers(Z)
        return h1
    

class Decoder_sharp_1024_v2(nn.Module):
    # no max pool used in encoder
    def __init__(self,d_latent = 64):
        """ 

        """
        super().__init__()
        self.d_latent = d_latent
        self.layers = nn.Sequential(
                                    nn.Linear(self.d_latent, 1024),
                                    nn.ReLU(),
                                    nn.Unflatten(1, (64, 4, 4)),
                                    nn.ConvTranspose2d(64, 32, kernel_size=3), #stride = 3),
                                    nn.ConvTranspose2d(32, 32, kernel_size=3),#, stride = 2),
                                    nn.ConvTranspose2d(32, 32, kernel_size=3, stride = 4),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(32, 16,kernel_size=3), #stride = 3),
                                    nn.ConvTranspose2d(16, 16,kernel_size=3),
                                    nn.ConvTranspose2d(16, 16,kernel_size=3, stride = 4),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(16, 8, kernel_size=5),#, stride = 2),
                                    nn.ConvTranspose2d(8, 8, kernel_size=5),#, stride = 2),
                                    nn.ConvTranspose2d(8, 8, kernel_size=5, stride = 2),
                                    nn.BatchNorm2d(8),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(8, 4, kernel_size=5),
                                    nn.ConvTranspose2d(4, 4, kernel_size=5),
                                    nn.ConvTranspose2d(4, 4, kernel_size=5),
                                    nn.BatchNorm2d(4),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(4, 1, kernel_size=5), #padding = 1),
                                    nn.ConvTranspose2d(1, 1, kernel_size=2), #padding = 1),
                                    nn.BatchNorm2d(1)
                                    )
    def forward(self, Z):
        h1 = self.layers(Z)
        return h1


class Encoder_sharp_1024_v2(nn.Module):
    
    def __init__(self, d_latent = 64):
        """ 
        Encoder using `nn.Module`. Series of 2d-Convolutions, dropouts and Relus

        """
        super().__init__()
        self.d_latent = d_latent
        self.layers = nn.Sequential(
                                    # nn.Conv2d(1, 8, kernel_size=3),
                                    # nn.ReLU(),
                                    #nn.MaxPool2d(2),
                                    nn.Conv2d(1, 8, kernel_size=5),
                                    nn.Conv2d(8, 8, kernel_size=5),
                                    nn.Conv2d(8, 8, kernel_size=5),
                                    #nn.MaxPool2d(2),
                                    nn.BatchNorm2d(8),
                                    nn.ReLU(),
                                    nn.Conv2d(8, 16, kernel_size=3),
                                    nn.Conv2d(16, 16, kernel_size=3),
                                    nn.Conv2d(16, 16, kernel_size=3),
                                    nn.MaxPool2d(2),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU(),
                                    nn.Conv2d(16, 32, kernel_size=3),
                                    nn.Conv2d(32, 32, kernel_size=3),
                                    nn.Conv2d(32, 32, kernel_size=3),
                                    nn.MaxPool2d(2),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.Conv2d(32, 64, kernel_size=3),
                                    nn.Conv2d(64, 64, kernel_size=3),
                                    nn.Conv2d(64, 64, kernel_size=3),
                                    nn.MaxPool2d(2),
                                    nn.BatchNorm2d(64),
                                    nn.Flatten(),
                                    nn.Linear(65536, d_latent),
                                    )
        self.sigmoid=nn.Sigmoid()

        self.layers_dec = nn.Sequential(
                            nn.Linear(self.d_latent, 65536),
                            nn.ReLU(),
                            nn.Unflatten(1, (64, 32, 32)),
                            nn.ConvTranspose2d(64, 32, kernel_size=3), #stride = 3),
                            nn.ConvTranspose2d(32, 32, kernel_size=3),#, stride = 2),
                            nn.ConvTranspose2d(32, 32, kernel_size=3, stride = 4),
                            nn.BatchNorm2d(32),
                            nn.ReLU(),
                            nn.ConvTranspose2d(32, 16,kernel_size=3), #stride = 3),
                            nn.ConvTranspose2d(16, 16,kernel_size=3),
                            nn.ConvTranspose2d(16, 16,kernel_size=3, stride = 4),
                            nn.BatchNorm2d(16),
                            nn.ReLU(),
                            nn.ConvTranspose2d(16, 8, kernel_size=5),#, stride = 2),
                            nn.ConvTranspose2d(8, 8, kernel_size=5),#, stride = 2),
                            nn.ConvTranspose2d(8, 8, kernel_size=5, stride = 2),
                            nn.BatchNorm2d(8),
                            nn.ReLU(),
                            nn.ConvTranspose2d(8, 4, kernel_size=5),
                            nn.ConvTranspose2d(4, 4, kernel_size=5),
                            nn.ConvTranspose2d(4, 4, kernel_size=5),
                            nn.ConvTranspose2d(4, 1, kernel_size=5), #padding = 1),
                            nn.ConvTranspose2d(1, 1, kernel_size=2), #padding = 1),
                            nn.BatchNorm2d(1)
                            )


    def forward(self, X):
        h1 = self.layers(X)
        h1 = self.sigmoid(h1)
        h1 = self.layers_dec(h1)
        return h1