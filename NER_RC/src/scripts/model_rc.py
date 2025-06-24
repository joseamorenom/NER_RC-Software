# -*- coding: utf-8 -*-
"""
Created on Thu May  4 11:19:59 2023

@author: gita
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import json
import os
import gc
from distutils.dir_util import copy_tree
import argparse
import flair
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, TransformerWordEmbeddings
from torch import nn, tanh, sigmoid, relu, FloatTensor, rand, stack, optim, cuda, softmax, save, device, tensor, int64, no_grad, concat
from flair.data import Sentence

default_path = os.path.dirname(os.path.abspath(__file__))
tagger_document = 0
embeddings = 0
json_data = 0
train_loader = 0
val_loader = 0
cnn = 0
optimizer = 0
criterion = 0
device = 0


class MyDataset(Dataset):
    def __init__(self, len_c1=7, len_c2=5, len_c3=11):
        global json_data
        
        def create_vector(c1,sentence):
            #print("Hola mundo")
            if len(c1): c1 = torch.cat([c1,sentence], dim=0)
            else: c1 = sentence
            return c1
        
        def fix_tensor(tensor, size):
            
            
            while tensor.shape[2] < size:
                tensor = torch.cat([tensor,torch.zeros(1,1,1,1024)], dim=2)
                
            tensor = tensor[:,:,:size,:]
            return tensor

        tensor_temp = torch.Tensor(json_data['flat_emb'])

        data = tensor_temp.reshape((tensor_temp.shape[0],1,-1,1024))

        
        self.targets =  create_vector(self.targets,torch.Tensor(json_data['relation']))
        
        

        for n_sen in range(tensor_temp.shape[0]):


            tensor_temp = data[n_sen,0,:json_data['h_pos'][n_sen][0],:].reshape((1, 1,-1,1024))
            self.c1 = create_vector(self.c1,fix_tensor(tensor_temp, len_c1))
            
            tensor_temp = data[n_sen,0,json_data['h_pos'][n_sen][0]:json_data['h_pos'][n_sen][-1]+1,:].mean(dim=0).reshape((1,1024))
            self.h1 = create_vector(self.h1,tensor_temp)
            
            tensor_temp = data[n_sen,0,json_data['h_pos'][n_sen][-1]+1:json_data['t_pos'][n_sen][0],:].reshape((1,1,-1,1024))
            self.c2 = create_vector(self.c2,fix_tensor(tensor_temp, len_c2))
            
            tensor_temp = data[n_sen,0,json_data['t_pos'][n_sen][0]:json_data['t_pos'][n_sen][-1]+1,:].mean(dim=0).reshape((1,1024))
            self.h2 = create_vector(self.h2,tensor_temp)                  
            
            tensor_temp = data[n_sen,0,json_data['t_pos'][n_sen][-1]+1:,:].reshape((1, 1,-1,1024))
            self.c3 = create_vector(self.c3,fix_tensor(tensor_temp, len_c3))
             
        del data
        del tensor_temp
        del json_data
        gc.collect()
        self.targets = self.targets.to(torch.int64)
        
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        c1x = self.c1[index]
        h1x = self.h1[index]
        c2x = self.c2[index]
        h2x = self.h2[index]
        c3x = self.c3[index]
        y = self.targets[index]
        return c1x,h1x,c2x,h2x,c3x, y
    
    
def update_step(c1, h1,c2,h2,c3, label):
    global cnn
    global optimizer
    global criterion
    prediction = cnn(c1, h1,c2,h2,c3)
    optimizer.zero_grad()
    loss = criterion(prediction, label)
    loss.backward()
    optimizer.step()
    acc = (nn.Softmax(dim=1)(prediction).detach().argmax(dim=1) == label).type(torch.float).sum().item()
    #print(acc)
    return loss.item(), acc

def evaluate_step(c1, h1,c2,h2,c3, label):
    global cnn
    global optimizer
    global criterion
    prediction = cnn(c1, h1,c2,h2,c3)
    loss = criterion(prediction, label)
    acc = (nn.Softmax(dim=1)(prediction).detach().argmax(dim=1) == label).type(torch.float).sum().item()
    return loss.item(), acc

def train_one_epoch(epoch):    
    global train_loader
    global val_loader
    global device
    if (device == torch.device('cuda:0')): cnn.cuda()
    train_loss, valid_loss, acc_train, acc_valid = 0.0, 0.0, 0.0, 0.0    
    for batch_idx, (c1, h1,c2,h2,c3, targets) in enumerate(train_loader):
        train_loss_temp, acc_train_temp = update_step(c1.to(device), h1.to(device),c2.to(device),h2.to(device),c3.to(device), targets.to(device))  
        train_loss += train_loss_temp
        acc_train += acc_train_temp
    for batch_idx, (c1, h1,c2,h2,c3, targets) in enumerate(val_loader):
        valid_loss_temp, acc_valid_temp = evaluate_step(c1.to(device), h1.to(device),c2.to(device),h2.to(device),c3.to(device), targets.to(device))
        valid_loss += valid_loss_temp
        acc_valid += acc_valid_temp
    # Guardar modelo si es el mejor hasta ahora
    global best_valid_loss
    if epoch % 10 == 0:
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save({'epoca': epoch,
                        'model_state_dict': cnn.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': valid_loss}, 
                       '/../../RC/model/best_model.pt')
    
    return train_loss/len(train_loader.dataset), valid_loss/len(val_loader.dataset), acc_train/len(train_loader.dataset), acc_valid/len(val_loader.dataset)


def FocalLoss(input, target, gamma=0, alpha=None, size_average=True):
    from torch.autograd import Variable
    if input.dim()>2:
        input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
        input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
        input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
    target = target.view(-1,1)

    logpt = nn.functional.log_softmax(input)
    logpt = logpt.gather(1,target)
    logpt = logpt.view(-1)
    pt = Variable(logpt.data.exp())

    if alpha is not None:
        if alpha.type()!=input.data.type():
            alpha = alpha.type_as(input.data)
        at = alpha.gather(0,target.data.view(-1))
        logpt = logpt * Variable(at)

    loss = -1 * (1-pt)**gamma * logpt
    if size_average: return loss.mean()
    else: return loss.sum()

    
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):

        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.early_stop = False

    def __call__(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.early_stop = False
            
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            print('Less')
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        

                


def SoftmaxModified(x):
    input_softmax = x.transpose(0,1)
    function_activation = nn.Softmax(dim=1)
    output = function_activation(input_softmax)
    output = output.transpose(0,1)
    return output


class MultiModalGMUAdapted(nn.Module):

    def __init__(self, input_size_array, hidden_size, dropoutProbability):
        """Initialize params."""
        super(MultiModalGMUAdapted, self).__init__()
        self.input_size_array = input_size_array
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropoutProbability)
        
        self.h_1_layer = nn.Linear(input_size_array[0], hidden_size, bias=False)
        self.h_2_layer = nn.Linear(input_size_array[1], hidden_size, bias=False)
        self.h_3_layer = nn.Linear(input_size_array[2], hidden_size, bias=False)
        self.h_4_layer = nn.Linear(input_size_array[3], hidden_size, bias=False)
        self.h_5_layer = nn.Linear(input_size_array[4], hidden_size, bias=False)
        
        self.z_1_layer = nn.Linear(input_size_array[0], hidden_size, bias=False)
        self.z_2_layer = nn.Linear(input_size_array[1], hidden_size, bias=False)
        self.z_3_layer = nn.Linear(input_size_array[2], hidden_size, bias=False)
        self.z_4_layer = nn.Linear(input_size_array[3], hidden_size, bias=False)
        self.z_5_layer = nn.Linear(input_size_array[4], hidden_size, bias=False)
        
        
        #self.z_weights = [nn.Linear(input_size_array[m], hidden_size, bias=False) for m in range(modalities_number)]
        #self.input_weights = [nn.Linear(size, hidden_size, bias=False) for size in input_size_array]

        
    def forward(self, inputModalities):
        """Propogate input through the network."""
        # h_modalities = [self.dropout(self.input_weights[i](i_mod)) for i,i_mod in enumerate(inputModalities)]
        # h_modalities = [tanh(h) for h in h_modalities]
        
        h1 = tanh(self.dropout(self.h_1_layer(inputModalities[0])))
        h2 = tanh(self.dropout(self.h_2_layer(inputModalities[1])))
        h3 = tanh(self.dropout(self.h_3_layer(inputModalities[2])))
        h4 = tanh(self.dropout(self.h_4_layer(inputModalities[3])))
        h5 = tanh(self.dropout(self.h_5_layer(inputModalities[4])))

        z1 = self.dropout(self.z_1_layer(inputModalities[0]))
        z2 = self.dropout(self.z_2_layer(inputModalities[1]))
        z3 = self.dropout(self.z_3_layer(inputModalities[2]))   
        z4 = self.dropout(self.z_4_layer(inputModalities[3]))
        z5 = self.dropout(self.z_5_layer(inputModalities[4]))
        

        #z_modalities = [self.dropout(self.z_weights[i](i_mod)) for i,i_mod in enumerate(inputModalities)]
        z_modalities = stack([z1, z2, z3, z4, z5])
        z_normalized = SoftmaxModified(z_modalities)
        final = z_normalized[0] * h1 + z_normalized[1] * h2 + z_normalized[2] * h3 + z_normalized[3] * h4 + z_normalized[4] * h5
        
    
        return final
    
class MyCNN(nn.Module):
    def __init__(self, num_classes=10, len_c1=7, len_c2=5, len_c3=11):
        super(MyCNN, self).__init__()
        shape1 = (((len_c1-2)))#-2)#//2)-2)//2)
        shape2 = (((len_c2-2)))#-2)#//2)-2)//2)
        shape3 = (((len_c3-2)))#-2)#//2)-2)//2)

        # Define convolutional layers
        self.conv_layers1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(shape1,1)),
        )
        
        self.conv_layers2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(shape2,1)),
        )
        
        self.conv_layers3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(shape3,1)),
        )
        

        self.multi_gmu = MultiModalGMUAdapted([1024,1024,1024,1024,1024], 1024, 0.5)

        

        

        
        self.fc_simple_layers_multi = nn.Sequential(
            nn.Linear(1024 , 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )


    def forward(self, c1, h1,c2,h2,c3):

        # Pass inputs through convolutional layers

        c1 = self.conv_layers1(c1)
        c2 = self.conv_layers2(c2)
        c3 = self.conv_layers3(c3)
        #print(c1.shape)
        
        h1 = tanh(h1)
        h2 = tanh(h2)
        #print(c1.shape)
        c1 = torch.flatten(c1, start_dim=1)
        c2 = torch.flatten(c2, start_dim=1)
        c3 = torch.flatten(c3, start_dim=1)
        #print(c1.shape)
        
        mgmu_out, mgmu_weigths = self.multi_gmu([c1,h1,c2,h2,c3]) 
      
        
        # Multi GMU
        x = self.fc_simple_layers_multi(mgmu_out)

        # Return final output
        return x
    
def define_model():
    global cnn
    global optimizer
    global criterion
    
    cnn = MyCNN()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
    criterion = lambda pred,tar: FocalLoss(input=pred,target=tar,gamma=0.7)
    
def train_model():
    max_epochs, best_valid_loss = 200, np.inf
    running_loss = np.zeros(shape=(max_epochs, 4))
    early_stopping = EarlyStopping(patience=10, min_delta=0.01)
    
    for epoch in range(max_epochs):
        running_loss[epoch] = train_one_epoch(epoch)
        early_stopping(running_loss[epoch, 1])
        print(f"Epoch {epoch} \t Train_loss = {running_loss[epoch, 0]:.4f} \t Valid_loss = {running_loss[epoch, 1]:.4f} \n\t\t\t Train_acc = {running_loss[epoch, 2]:.4f} \t Valid_acc = {running_loss[epoch, 3]:.4f}")
        if early_stopping.early_stop:
          print("We are at epoch:", epoch)
          break
      
        
def usage_cuda_rc(cuda):
    global device
    if cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        flair.device = device
        if flair.device == torch.device('cpu'): return 'Error handling GPU, CPU will be used'
        elif flair.device == torch.device('cuda:0'): return 'GPU detected, GPU will be used'
    else:
        device = torch.device('cpu')
        flair.device = device
        return 'CPU will be used'
    
    
def create_embbedings():
    global embeddings
    if (not embeddings):
        embeddings = TransformerWordEmbeddings(
                    model='xlm-roberta-large',
                    layers="-1",
                    subtoken_pooling="first", 
                    fine_tune=True,
                    use_context=True,
                )



    
def prepare_data():
    create_embbedings()
    global embeddings
    global json_data
    #Embbeb data
    
    path_files = default_path + '/../../data/RC/'
    
    rel2id_file = path_files + 'rel2id.json' 
    with open(rel2id_file, mode='r') as f: 
        rel2id = json.load(f)
        
        
    path_data = path_files+"train.txt"

    #Json to save the data
    json_data = {'flat_emb':[], 'relation':[], 'h_pos':[], 't_pos':[]}
    PADDING = np.zeros(1024)
    doc=0
    with open(path_data, mode='r', encoding='utf-8') as f:
        sentence_temp = []
        h_pos = []
        t_pos = []
        current_ent=''
        cont=0
        
        for n,line in enumerate(f.readlines()):
            if line != '\n':
                sentence_temp.append(line.split('\t')[0])
                
                if line.split('\t')[1] != 'O':
                    if current_ent == '':
                        h_pos.append(cont)
                        current_ent = line.split('\t')[1]
                        
                    elif line.split('\t')[1] == current_ent:
                        h_pos.append(cont)
                        
                    else:
                        t_pos.append(cont)
                        
                if line.split('\t')[2].replace('\n','') != '-' : relation = line.split('\t')[2].replace('\n','') 

                cont += 1
                
            else:
                
                #Embbedding sentence
                sentence = Sentence(sentence_temp)
                embeddings.embed(sentence)
                
                

                sentence_emb_flatten = []
                for tk in sentence: 
                    #flatten_embeddings    
                    if len(sentence_emb_flatten): sentence_emb_flatten = np.hstack((sentence_emb_flatten,
                                                                               tk.embedding.detach().to('cpu').numpy()))
                    else: sentence_emb_flatten = tk.embedding.detach().to('cpu').numpy()
                       
                number_padding = 100 - len(sentence)
                
                if number_padding > 0: 
                    for pd in range(number_padding):
                        sentence_emb_flatten = np.hstack((sentence_emb_flatten,
                                                          PADDING))
                       
                #Save embeddings information
                json_data['flat_emb'].append(list(sentence_emb_flatten))
                json_data['h_pos'].append(h_pos)
                json_data['t_pos'].append(t_pos)
                json_data['relation'].append(rel2id[relation])
                
                sentence_temp = []
                h_pos = []
                t_pos = []
                current_ent=''
                cont=0
    dataset = MyDataset()

    train_set_size = int(len(dataset) * 0.9)
    valid_set_size = len(dataset) - train_set_size
        
    train_dataset, val_dataset = random_split(dataset, [train_set_size, valid_set_size ])
    del dataset
    global train_loader 
    global val_loader
                                                 
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)