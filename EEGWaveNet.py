#Import Libraries
import matplotlib.pyplot as plt
import os
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy import signal
from sklearn.utils.class_weight import compute_class_weight as classweight
import time

#Import Libraries for NN 
import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, BCELoss, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD

class Functions:
    def EEGWaveNet(n_chans,n_classes):
        
        class Net(nn.Module):
            
            ###########################################################################################################
            ## Model is tuned for seizure classification from EEG in CHB-MIT Dataset, further use require new tuning ##
            ###########################################################################################################
            
            def __init__(self):
                super(Net, self).__init__()

            ###########################################################################################################

                self.temp_conv1 = nn.Conv2d(n_chans, n_chans, kernel_size=(1,2), stride=2 ,groups=n_chans)
                self.temp_conv2 = nn.Conv2d(n_chans, n_chans, kernel_size=(1,2), stride=2 ,groups=n_chans)
                self.temp_conv3 = nn.Conv2d(n_chans, n_chans, kernel_size=(1,2), stride=2 ,groups=n_chans)
                self.temp_conv4 = nn.Conv2d(n_chans, n_chans, kernel_size=(1,2), stride=2 ,groups=n_chans)
                self.temp_conv5 = nn.Conv2d(n_chans, n_chans, kernel_size=(1,2), stride=2 ,groups=n_chans)
                self.temp_conv6 = nn.Conv2d(n_chans, n_chans, kernel_size=(1,2), stride=2 ,groups=n_chans)

            ###########################################################################################################

                self.chpool1    = Sequential(
                    nn.Conv2d(n_chans, 32, kernel_size=(1,4),groups=1),
                    torch.nn.BatchNorm2d(32),
                    nn.LeakyReLU(0.01),
                    nn.Conv2d(32, 32, kernel_size=(1,4),groups=1),
                    torch.nn.BatchNorm2d(32),
                    nn.LeakyReLU(0.01))

                self.chpool2    = Sequential(
                    nn.Conv2d(n_chans, 32, kernel_size=(1,4),groups=1),
                    torch.nn.BatchNorm2d(32),
                    nn.LeakyReLU(0.01),
                    nn.Conv2d(32, 32, kernel_size=(1,4),groups=1),
                    torch.nn.BatchNorm2d(32),
                    nn.LeakyReLU(0.01))

                self.chpool3    = Sequential(
                    nn.Conv2d(21, 32, kernel_size=(1,4),groups=1),
                    torch.nn.BatchNorm2d(32),
                    nn.LeakyReLU(0.01),
                    nn.Conv2d(32, 32, kernel_size=(1,4),groups=1),
                    torch.nn.BatchNorm2d(32),
                    nn.LeakyReLU(0.01))
                    
                self.chpool4    = Sequential(
                    nn.Conv2d(n_chans, 32, kernel_size=(1,4),groups=1),
                    torch.nn.BatchNorm2d(32),
                    nn.LeakyReLU(0.01),
                    nn.Conv2d(32, 32, kernel_size=(1,4),groups=1),
                    torch.nn.BatchNorm2d(32),
                    nn.LeakyReLU(0.01))

                self.chpool5    = Sequential(
                    nn.Conv2d(n_chans, 32, kernel_size=(1,4),groups=1),
                    torch.nn.BatchNorm2d(32),
                    nn.LeakyReLU(0.01),
                    nn.Conv2d(32, 32, kernel_size=(1,4),groups=1),
                    torch.nn.BatchNorm2d(32),
                    nn.LeakyReLU(0.01))

                self.classifier = Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(160,64),
                    nn.LeakyReLU(0.01),
                    nn.Dropout(p=0.2),
                    nn.Linear(64,32),
                    nn.Sigmoid(),
                    nn.Linear(32,n_classes))

            def forward(self, x , training=True):

                temp_x  = self.temp_conv1(x)               
                temp_w1 = self.temp_conv2(temp_x)         
                temp_w2 = self.temp_conv3(temp_w1)      
                temp_w3 = self.temp_conv4(temp_w2)       
                temp_w4 = self.temp_conv5(temp_w3)      
                temp_w5 = self.temp_conv6(temp_w4)      

            ###########################################################################################################

                w1      = self.chpool1(temp_w1).mean(dim=(-2,-1))
                w2      = self.chpool2(temp_w2).mean(dim=(-2,-1))
                w3      = self.chpool3(temp_w3).mean(dim=(-2,-1))
                w4      = self.chpool4(temp_w4).mean(dim=(-2,-1))
                w5      = self.chpool5(temp_w5).mean(dim=(-2,-1))

            ###########################################################################################################
            
                concat_vector   = torch.cat([w1,w2,w3,w4,w5],1)
                classes         = F.log_softmax(self.classifier(concat_vector),dim=1)  

                return(classes)
            
        return(Net().float())
    
    def train(Model,X_train,y_train,X_val,y_val,n_epochs,batch_size,learning_rate,weight_decay,patience,n_classes):
        
        Saved_model = Model
        Wait = 0
               
        List_train_loss       = []
        List_val_loss         = []
        List_val_acc          = []
        T                     = []
               
        optimizer             = Adam(Model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        weights               = classweight(class_weight="balanced",classes=np.arange(n_classes),y=y_train.numpy())
        class_weights         = torch.FloatTensor(weights).cuda()
        CE_loss_func          = CrossEntropyLoss(weight=class_weights)
        val_weights           = classweight(class_weight="balanced",classes=np.arange(n_classes),y=y_val.numpy())        
        val_class_weights     = torch.FloatTensor(val_weights).cuda()
        val_CE_loss_func      = CrossEntropyLoss(weight=val_class_weights)
        trainset              = [[X_train[i],y_train[i]] for i in range(X_train.size()[0])]
        trainloader           = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        valset                = [[X_val[i],y_val[i]] for i in range(X_val.size()[0])]
        valloader             = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True)
        
        Model.cuda()
        
        for epoch in range(0,n_epochs):
            
            t0 = time.time()
            
            for batch_idx, (data, target) in enumerate(trainloader):

                Model.train()
                
                output     = Model(data.float().cuda())[1]

                optimizer.zero_grad()
                train_loss = CE_loss_func(output, target.cuda())
                train_loss.backward()       
                optimizer.step()
                
            lossall = []
            accall  = []
            f1all   = []
                
            with torch.no_grad():
                
                for batch_idx, (data, target) in enumerate(valloader):

                    output                     = Model(data.float().cuda())      
                    val_loss                   = val_CE_loss_func(output, target.cuda())

                    val_predictions            = np.argmax(list(output.cpu().numpy()), axis=1)
                    val_acc                    = accuracy_score(target.cpu(), val_predictions)*100
                    
                    lossall.append(val_loss)
                    accall.append(val_acc)

                val_loss  = torch.mean(torch.tensor(lossall))
                val_acc   = np.average(accall)

            print("EPOCH : ",epoch)
            print("train loss = ","{:.5f}".format(train_loss.item()))
            print("val loss = ","{:.5f}".format(val_loss.item()))
            print("acc = ",val_acc)
            print("Elapsed Time per epoch = ", "{:.5f}".format(time.time()-t0), "s")
            print("===================================================================================\n")
            
                        
            T.append(time.time()-t0)

            if epoch>9:
                if val_loss.item()<=np.min(List_val_loss):
                    Saved_model = Model
                    Wait = 0
                else:
                    Wait += 1

            List_train_loss.append(train_loss.item())
            List_val_loss.append(val_loss.item())       
            List_val_acc.append(val_acc)
            List_val_f_one.append(val_f_one)

            if Wait >= patience:
                break
            
            history = {"train loss"          : List_train_loss,
                      "val loss"             : List_val_loss,
                      "accuracies"           : List_val_acc,
                      "f1 scores"            : List_val_f_one}

                
        return(Saved_model,history,T)
