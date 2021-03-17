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

#Import Libraries for NN and XAI
import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, BCELoss, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, Conv1d, MaxPool1d, BatchNorm1d
from torch.optim import Adam, SGD

#Import Libraries for DML
from pytorch_metric_learning import losses, miners, distances, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

class Functions:
    def build(n_chans,n_classes):
        class Net(nn.Module):
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
                    nn.Conv2d(21, 128, kernel_size=(1,4),groups=1),
                    torch.nn.BatchNorm2d(128),
                    nn.LeakyReLU(0.01),
                    nn.Conv2d(128, 64, kernel_size=(1,4),groups=1),
                    torch.nn.BatchNorm2d(64),
                    nn.LeakyReLU(0.01),
                    nn.Conv2d(64, 32, kernel_size=(1,4),groups=1),
                    torch.nn.BatchNorm2d(32),
                    nn.LeakyReLU(0.01))

                self.chpool2    = Sequential(
                    nn.Conv2d(21, 128, kernel_size=(1,4),groups=1),
                    torch.nn.BatchNorm2d(128),
                    nn.LeakyReLU(0.01),
                    nn.Conv2d(128, 64, kernel_size=(1,4),groups=1),
                    torch.nn.BatchNorm2d(64),
                    nn.LeakyReLU(0.01),
                    nn.Conv2d(64, 32, kernel_size=(1,4),groups=1),
                    torch.nn.BatchNorm2d(32),
                    nn.LeakyReLU(0.01))

                self.chpool3    = Sequential(
                    nn.Conv2d(21, 128, kernel_size=(1,4),groups=1),
                    torch.nn.BatchNorm2d(128),
                    nn.LeakyReLU(0.01),
                    nn.Conv2d(128, 64, kernel_size=(1,4),groups=1),
                    torch.nn.BatchNorm2d(64),
                    nn.LeakyReLU(0.01),
                    nn.Conv2d(64, 32, kernel_size=(1,4),groups=1),
                    torch.nn.BatchNorm2d(32),
                    nn.LeakyReLU(0.01))
                    
                self.chpool4    = Sequential(
                    nn.Conv2d(21, 128, kernel_size=(1,4),groups=1),
                    torch.nn.BatchNorm2d(128),
                    nn.LeakyReLU(0.01),
                    nn.Conv2d(128, 64, kernel_size=(1,4),groups=1),
                    torch.nn.BatchNorm2d(64),
                    nn.LeakyReLU(0.01),
                    nn.Conv2d(64, 32, kernel_size=(1,4),groups=1),
                    torch.nn.BatchNorm2d(32),
                    nn.LeakyReLU(0.01))

                self.chpool5    = Sequential(
                    nn.Conv2d(21, 128, kernel_size=(1,4),groups=1),
                    torch.nn.BatchNorm2d(128),
                    nn.LeakyReLU(0.01),
                    nn.Conv2d(128, 64, kernel_size=(1,4),groups=1),
                    torch.nn.BatchNorm2d(64),
                    nn.LeakyReLU(0.01),
                    nn.Conv2d(64, 32, kernel_size=(1,4),groups=1),
                    torch.nn.BatchNorm2d(32),
                    nn.LeakyReLU(0.01))

                self.classifier = Sequential(
                    nn.Linear(160,64),
                    nn.LeakyReLU(0.01),
                    nn.Linear(64,32),
                    nn.Sigmoid(),
                    nn.Linear(32,n_classes))

            def forward(self, x , training=True):

                temp_x          = self.temp_conv1(x)               #128 hz , output_size = 512
                temp_gamma      = self.temp_conv2(temp_x)          #64 hz  , output_size = 256
                temp_beta       = self.temp_conv3(temp_gamma)      #32 hz  , output_size = 128
                temp_alpha      = self.temp_conv4(temp_beta)       #16 hz  , output_size = 64
                temp_delta      = self.temp_conv5(temp_alpha)      #8 hz   , output_size = 32
                temp_theta      = self.temp_conv6(temp_delta)      #4 hz   , output_size = 16

            ###########################################################################################################

                gamma           = self.chpool1(temp_gamma).mean(dim=(-2,-1))
                beta            = self.chpool2(temp_beta).mean(dim=(-2,-1))
                alpha           = self.chpool3(temp_alpha).mean(dim=(-2,-1))
                delta           = self.chpool4(temp_delta).mean(dim=(-2,-1))
                theta           = self.chpool5(temp_theta).mean(dim=(-2,-1))

            ###########################################################################################################
            
                embeddings      = torch.cat([theta,delta,alpha,beta,gamma],1)
                classes         = F.log_softmax(self.classifier(embeddings),dim=1)  

                return(embeddings, classes)
            
        return(Net().float())
    
    def DML_train(Model,X_train,y_train,X_val,y_val,n_epochs,batch_size,learning_rate,weight_decay,patience,n_classes):
        
        Saved_model = Model
        Wait = 0
               
        List_train_loss       = []
        List_val_loss         = []
        List_val_acc          = []
        List_val_f_one        = []
               
        optimizer             = Adam(Model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        contrastive_loss_func = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
        weights               = classweight(class_weight="balanced",classes=np.arange(n_classes),y=y_train.numpy())        
        class_weights         = torch.FloatTensor(weights).cuda()
        CE_loss_func          = CrossEntropyLoss(weight=class_weights)
                
        trainset              = [[X_train[i],y_train[i]] for i in range(X_train.size()[0])]
        trainloader           = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        valset                = [[X_val[i],y_val[i]] for i in range(X_val.size()[0])]
        valloader             = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True)
        
        Model = Model.cuda()
        
        for epoch in range(0,n_epochs):
            
            t0 = time.time()
            
            for batch_idx, (data, target) in enumerate(trainloader):

                Model.train()

                optimizer.zero_grad()
                embeddings, output   = Model(data.float().cuda()) 
                emb_train_loss       = contrastive_loss_func(embeddings, target)
                cls_train_loss       = CE_loss_func(output, target.cuda())
                train_loss           = emb_train_loss*0.1+cls_train_loss*1
                train_loss.backward()       
                optimizer.step()
                
            lossall = []
            accall  = []
            f1all   = []
                
            with torch.no_grad():
                
                for batch_idx, (data, target) in enumerate(valloader):

                    embeddings, output         = Model(data.float().cuda())      
                    emb_val_loss               = contrastive_loss_func(embeddings, target)
                    cls_val_loss               = CE_loss_func(output, target.cuda())
                    val_loss                   = emb_val_loss*0.1+cls_val_loss*1

                    val_predictions            = np.argmax(list(output.cpu().numpy()), axis=1)
                    val_acc                    = accuracy_score(target.cpu(), val_predictions)*100
                    val_f_one                  = f1_score(target.cpu(), val_predictions, average='macro')
                    
                    lossall.append(val_loss)
                    accall.append(val_acc)
                    f1all.append(val_f_one)  
                    
                val_loss  = torch.mean(torch.tensor(lossall))
                val_acc   = np.average(accall)
                val_f_one = np.average(f1all)

            print("EPOCH : ",epoch)
            print("train loss = ","{:.5f}".format(train_loss.item()), "\t emb_train loss = ","{:.5f}".format(emb_train_loss.item()), "\t cls_train loss = ","{:.5f}".format(cls_train_loss.item()))
            print("val loss = ","{:.5f}".format(val_loss.item()))
            print("acc = ",val_acc)
            print("f1 = ",val_f_one)
            print("Elapsed Time = ", int(time.time()-t0), "s")
            print("===================================================================================\n")

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

                
        return(Saved_model,history)
    
    def regular_train(Model,X_train,y_train,X_val,y_val,n_epochs,batch_size,learning_rate,weight_decay,patience,n_classes):
        
        Saved_model = Model
        Wait = 0
               
        List_train_loss       = []
        List_val_loss         = []
        List_val_acc          = []
        List_val_f_one        = []
               
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

                    output                     = Model(data.float().cuda())[1]      
                    val_loss                   = val_CE_loss_func(output, target.cuda())

                    val_predictions            = np.argmax(list(output.cpu().numpy()), axis=1)
                    val_acc                    = accuracy_score(target.cpu(), val_predictions)*100
                    val_f_one                  = f1_score(target.cpu(), val_predictions, average='macro')
                    
                    lossall.append(val_loss)
                    accall.append(val_acc)
                    f1all.append(val_f_one)  

                val_loss  = torch.mean(torch.tensor(lossall))
                val_acc   = np.average(accall)
                val_f_one = np.average(f1all)

            print("EPOCH : ",epoch)
            print("train loss = ","{:.5f}".format(train_loss.item()))
            print("val loss = ","{:.5f}".format(val_loss.item()))
            print("acc = ",val_acc)
            print("f1 = ",val_f_one)
            print("Elapsed Time = ", int(time.time()-t0), "s")
            print("===================================================================================\n")

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

                
        return(Saved_model,history)
    
    def evaluate(Model,X_test,y_test):
        with torch.no_grad():
            test_embeddings, output = Model(X_test.float())[0], Model(X_test.float())[1]

        prob = list(output.numpy())
        test_predictions = np.argmax(prob, axis=1)
        test_acc = accuracy_score(y_test, test_predictions)*100
        test_f_one = f1_score(y_test, test_predictions, average='macro')
        print("test_acc = ",test_acc)
        print("test_f1 = ",test_f_one)
        
        return(test_embeddings,test_acc,test_f_one)
    
    def clustering(embeddings,label,classes):
        embeddings_embedded = TSNE(n_components=len(classes)).fit_transform(embeddings)
        for i in range(len(classes)):
            x = [embeddings_embedded[:,0][j] for j in range(label.shape[0]) if label[j]==i]
            y = [embeddings_embedded[:,1][j] for j in range(label.shape[0]) if label[j]==i]
            
            plt.scatter(x,y,label=classes[i])
            
        plt.legend()
            
        return(embeddings_embedded)
    
    def norm(array):
        return np.array([((array[i]-np.min(array[i]))/(np.max(array[i])-np.min(array[i]))-0.5) for i in range(array.shape[0])])
