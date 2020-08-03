# -*- coding: utf-8 -*-
"""DynamicTile.ipynb


Original file is located at
    https://colab.research.google.com/drive/12iXCPWH5-vati6LxMfZuL8ifT6-9d-f3
"""

from google.colab import drive
drive.mount('/content/gdrive')

!pip install -q -U git+https://github.com/albumentations-team/albumentations   #LATEST ALBUMATIONS

import albumentations
albumentations.__version__

!pip install -q torchsummary
from torchsummary import summary

from google.colab import files
files.upload() #upload train.csv

from google.colab import files
files.upload() #upload train.csv

!pip install -q kaggle
import os
os.environ['KAGGLE_USERNAME'] = "se***" 
os.environ['KAGGLE_KEY'] = "***************************"

!kaggle datasets list -s efficientnet-pytorch

!kaggle datasets download -d hmendonca/efficientnet-pytorch 
import zipfile
with zipfile.ZipFile('efficientnet-pytorch.zip', 'r') as zip_ref:
    zip_ref.extractall('efficientnet-pytorch')
!kaggle datasets download -d gabrichy/nvidiaapex 

with zipfile.ZipFile('nvidiaapex.zip', 'r') as zip_ref:
    zip_ref.extractall('nvidiaapex')

!kaggle datasets download -d lopuhin/panda-2020-level-1-2 
with zipfile.ZipFile('panda-2020-level-1-2.zip', 'r') as zip_ref:
    zip_ref.extractall('panda-2020-level-1-2')

DEBUG = False

!pip install -q git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git

import os
import sys
sys.path = [
    'efficientnet-pytorch/EfficientNet-PyTorch/EfficientNet-PyTorch-master',
] + sys.path

!pip install -q efficientnet_pytorch

import time
import os
import skimage.io
import numpy as np
import pandas as pd
import cv2
import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler, SequentialSampler
from warmup_scheduler import GradualWarmupScheduler
from efficientnet_pytorch import model as enet
import albumentations
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm_notebook as tqdm
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


data_dir = 'panda-2020-level-1-2'
#df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
df_train = pd.read_csv('train.csv')
image_folder = os.path.join(data_dir, 'train_images/train_images')

kernel_type = 'model_july18'

enet_type = 'efficientnet-b4'
fold = 0

batch_size = 4
num_workers = 4
out_dim = 5
init_lr = 3e-4
warmup_factor = 10

warmup_epo = 1
n_epochs = 1 if DEBUG else 30
df_train = df_train.sample(100).reset_index(drop=True) if DEBUG else df_train

#device = torch.device('cuda')

print(image_folder)

kernel_type = 'model_july18'

image_idss = [

  '3385a0f7f4f3e7e7b380325582b115c9', #not includ very small
  '60502735df319ec33f862579fb6563d5', #not includ very small
  '41a2f4193aa1effcd56bac44e7affb03',  #not includ very small
  'eb88414bf869f5f93e3990b0d08c3bf3', #not includ very small
  '040b2c98538ec7ead1cbd6daacdb3f64', #- 1088 weird color
  '3790f55cad63053e956fb73027179707', #totally blank
 

]

for i in range(len(image_idss)):
  df_train = df_train[df_train['image_id'] != image_idss[i]]

df_train = df_train.reset_index()
df_train

skf = StratifiedKFold(5, shuffle=True, random_state=42)
df_train['fold'] = -1
for i, (train_idx, valid_idx) in enumerate(skf.split(df_train, df_train['isup_grade'])):
    df_train.loc[valid_idx, 'fold'] = i
df_train

pretrained_model = {''
    'efficientnet-b0': 'efficientnet-pytorch/efficientnet-b0-08094119.pth',   # 5 million parameters
    'efficientnet-b1': 'efficientnet-pytorch/efficientnet-b1-dbc7070a.pth',   # 7.7 million parameters
    'efficientnet-b2': 'efficientnet-pytorch/efficientnet-b2-27687264.pth',   # 9 million parameters
    'efficientnet-b3': 'efficientnet-pytorch/efficientnet-b3-c8376fa2.pth',   # 12 million parameters
    'efficientnet-b4': 'efficientnet-pytorch/efficientnet-b4-e116e8b3.pth',   # 19 million parameters
    'efficientnet-b5': 'efficientnet-pytorch/efficientnet-b5-586e6cc6.pth',   # 30 million parameters
    'efficientnet-b6': 'efficientnet-pytorch/efficientnet-b6-c76e70fd.pth',   # 43 million parameters
    'efficientnet-b7': 'efficientnet-pytorch/efficientnet-b7-dcc49843.pth'    # 66 million parameters

}

enet_type = 'efficientnet-b0'

class enetv2(nn.Module):
    def __init__(self, backbone, out_dim):
        super(enetv2, self).__init__()
        self.enet = enet.EfficientNet.from_name(backbone)
        self.enet.load_state_dict(torch.load(pretrained_model[backbone]))

        self.myfc = nn.Linear(self.enet._fc.in_features, out_dim)
        self.enet._fc = nn.Identity()

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(x)
        return x

import math

N_SIZE = 256
TARGET_IMAGE_H = 768
TARGET_IMAGE_W = 768


def getEliminationIds(images):
    badIdList = []
    N = len(images)
    if (N>0):
      h = images[0].shape[0]
      w = images[0].shape[1]
      c = images[0].shape[2]
      t = h*w*c
      for i in range(N):
          img = images[i]
          counthaha = 0
          for hh in range(h):
              for ww in range(w):
                  red = int(img[hh,ww,0])
                  green = int(img[hh,ww,1])
                  blue = int(img[hh,ww,2])
                  
                  if(abs(red-green)< 8 and abs(green-blue)< 8 and abs(red-blue)< 8):
                      counthaha +=1
                    
          rattio = counthaha/(h*w)
          #print(rattio)
          if(rattio>0.8):
              #print("80% or more empty", rattio)
              badIdList.append(i)
            
       
    #print("bad image ids ", badIdList)   
    return badIdList   

def getN(image,N_SIZE):
    h = image.shape[0]
    w = image.shape[1]
    c = image.shape[2]
    t = h*w*c
    #print("tot_pixel ", t)
    
    unique, counts = np.unique(image, return_counts=True)
    vals = dict(zip(unique, counts))
    total = 0
    for key, value in vals.items():
        total += value
    #print(h*w/(tile_size*tile_size))
    #print(h,w)
    #print(vals)
    #print("tot_pixel ", t," calculated ",total)
    if(255 in vals.keys()):
        ratio = (total-vals[255])/total
    else:
        ratio = 1
    #print("ratio ", ratio )
    N = int((h*w/N_SIZE**2)*ratio)
    #print("N " , N)
    return N



def tile(img, mask):
    
    N = getN(img,N_SIZE)
    ##############################print("N=" , N) DEBUG OPEN
    result = []
    shape = img.shape
    pad0,pad1 = (N_SIZE - shape[0]%N_SIZE)%N_SIZE, (N_SIZE - shape[1]%N_SIZE)%N_SIZE
    img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],constant_values=255)
    #mask = np.pad(mask,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],constant_values=0)

    img = img.reshape(img.shape[0]//N_SIZE,N_SIZE,img.shape[1]//N_SIZE,N_SIZE,3)
    img = img.transpose(0,2,1,3,4).reshape(-1,N_SIZE,N_SIZE,3)
    #mask = mask.reshape(mask.shape[0]//image_size,image_size,mask.shape[1]//image_size,image_size,3)
    #mask = mask.transpose(0,2,1,3,4).reshape(-1,image_size,image_size,3)
    if len(img) < N:
        #mask = np.pad(mask,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=0)
        img = np.pad(img,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=255)
    idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:N] 
    
   
    
    img = img[idxs]
    #mask = mask[idxs]
    
    bad_ids = getEliminationIds(img)
    
    
    for i in range(len(img)):
        if(not i in bad_ids):
            result.append({'img':img[i],'idx':i, 'N':N})
            #result.append({'img':img[i], 'mask':mask[i], 'idx':i, 'N':N})
        
    ###################print(N,"---->",len(result))  DEBUG OPEN
    return result


grid_size = 3
class PANDADataset(Dataset):
    def __init__(self, df, N_SIZE, tile_mode=0, rand=False, transform=None, transform_tile = None):

        self.df = df.reset_index(drop=True)
        self.N_SIZE = N_SIZE
    
        self.tile_mode = tile_mode
        self.rand = rand
        self.transform = transform
        self.transform_tile = transform_tile

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_id = row.image_id
        
        image = os.path.join(image_folder, f'{img_id}_1.jpeg')
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tiles = tile(image,None)
        N = len(tiles)
        frac, whole = math.modf(np.sqrt(N))

        if (frac >= 0.5):n_row_tiles = whole+1
        else:n_row_tiles = whole 
        #print(frac, whole, N, n_row_tiles)

        n_row_tiles = int(np.sqrt(N))
        images = np.zeros((N_SIZE * n_row_tiles, N_SIZE * n_row_tiles, 3))
      


        if self.rand:
            idxes = np.random.choice(list(range(N)), N, replace=False)
        else:
            idxes = list(range(N))

        for h in range(n_row_tiles):
          for w in range(n_row_tiles):
            i = h * n_row_tiles + w

            if len(tiles) > idxes[i]:
              this_img = tiles[idxes[i]]['img']

            else:
              this_img = np.ones((self.N_SIZE, self.N_SIZE, 3)).astype(np.uint8) * 255

            this_img = 255 - this_img

            if self.transform_tile is not None:
              this_img = self.transform_tile(image=this_img)['image']

            h1 = h * N_SIZE
            w1 = w * N_SIZE
            images[h1:h1+N_SIZE, w1:w1+N_SIZE] = this_img



     
        try : 
          images = cv2.resize(images,(TARGET_IMAGE_W,TARGET_IMAGE_H))

        except:
          images = np.zeros((TARGET_IMAGE_W,TARGET_IMAGE_H,3))
          print(' ERROR', img_id)

        grid_size = n_row_tiles
        
        if self.transform is not None:
            images = self.transform(image=images)['image']

        images = images.astype(np.float32)
        images /= 255
        images = images.transpose(2, 0, 1)

        label = np.zeros(5).astype(np.float32)
        label[:row.isup_grade] = 1.
        return torch.tensor(images), torch.tensor(label)

transforms_train = albumentations.Compose([
    albumentations.Transpose(p=0.5),
    albumentations.VerticalFlip(p=0.5),
    albumentations.HorizontalFlip(p=0.5),
    albumentations.RandomGridShuffle(grid=(grid_size, grid_size), always_apply=False, p=0.5)

    
])
transforms_val = albumentations.Compose([])



transforms_train_tiles = albumentations.Compose([
    albumentations.Transpose(p=0.5),
    albumentations.VerticalFlip(p=0.5),
    albumentations.HorizontalFlip(p=0.5),
    #albumentations.Rotate(value=1, border_mode=0)
   
    
    
])
transforms_val_tiles = albumentations.Compose([])

dataset_show = PANDADataset(df_train, N_SIZE, 0, transform=transforms_train,transform_tile = transforms_train_tiles )
from pylab import rcParams
rcParams['figure.figsize'] = 20,10
for i in range(2):
    f, axarr = plt.subplots(1,5)
    for p in range(5):
        idx = np.random.randint(0, len(dataset_show))
        img, label = dataset_show[idx]
        axarr[p].imshow(1. - img.transpose(0, 1).transpose(1,2).squeeze())
        axarr[p].set_title(str(sum(label)))

criterion = nn.BCEWithLogitsLoss()

def train_epoch(loader, optimizer):

    model.train()
    train_loss = []
    bar = tqdm(loader)
    for (data, target) in bar:
        
        data, target = data.to(device), target.to(device)
        loss_func = criterion
        optimizer.zero_grad()
        logits = model(data)
        loss = loss_func(logits, target)
        loss.backward()
        optimizer.step()

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smth: %.5f' % (loss_np, smooth_loss))
    return train_loss


def val_epoch(loader, get_output=False):

    model.eval()
    val_loss = []
    LOGITS = []
    PREDS = []
    TARGETS = []

    with torch.no_grad():
        for (data, target) in tqdm(loader):
            data, target = data.to(device), target.to(device)
            logits = model(data)

            loss = criterion(logits, target)

            pred = logits.sigmoid().sum(1).detach().round()
            LOGITS.append(logits)
            PREDS.append(pred)
            TARGETS.append(target.sum(1))

            val_loss.append(loss.detach().cpu().numpy())
        val_loss = np.mean(val_loss)

    LOGITS = torch.cat(LOGITS).cpu().numpy()
    PREDS = torch.cat(PREDS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()
    acc = (PREDS == TARGETS).mean() * 100.
    
    qwk = cohen_kappa_score(PREDS, TARGETS, weights='quadratic')
    qwk_k = cohen_kappa_score(PREDS[df_valid['data_provider'] == 'karolinska'], df_valid[df_valid['data_provider'] == 'karolinska'].isup_grade.values, weights='quadratic')
    qwk_r = cohen_kappa_score(PREDS[df_valid['data_provider'] == 'radboud'], df_valid[df_valid['data_provider'] == 'radboud'].isup_grade.values, weights='quadratic')
    print('qwk', qwk, 'qwk_k', qwk_k, 'qwk_r', qwk_r)

    if get_output:
        return LOGITS
    else:
        return val_loss, acc, qwk

device

from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
train_idx = np.where((df_train['fold'] != fold))[0]
valid_idx = np.where((df_train['fold'] == fold))[0]

df_this  = df_train.loc[train_idx]
df_valid = df_train.loc[valid_idx]#
#ratio = int(len(train_df)*0.8)
#df_this  = train_df
#f_valid = test_df

dataset_train = PANDADataset(df_this , N_SIZE, transform=transforms_train, transform_tile = transforms_train_tiles)
dataset_valid = PANDADataset(df_valid, N_SIZE, transform=transforms_val, transform_tile = transforms_val_tiles)

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, sampler=RandomSampler(dataset_train), num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, sampler=SequentialSampler(dataset_valid), num_workers=num_workers)

model = enetv2(enet_type, out_dim=out_dim)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=init_lr/warmup_factor)
#scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2, verbose=True, eps=1e-8)
scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs-warmup_epo)
scheduler = GradualWarmupScheduler(optimizer, multiplier=warmup_factor, total_epoch=warmup_epo, after_scheduler=scheduler_cosine)

print(len(dataset_train), len(dataset_valid))

m_path = 'model_july17_Best_fold0_lastEpoch_1_kappa_0.8227108037800486.pth'
model.load_state_dict(torch.load(m_path, map_location=torch.device(device)))

import os
#new_Folder = 'model'
new_Folder = '/content/gdrive/My Drive/Panda/Dynamic_256__target_768_b0_day4'
if(not os.path.exists(new_Folder)):
  os.makedirs(new_Folder)

start_epoch = 1 # Normally 1 
n_epochs = 30 # total = 31

qwk_max = 0.
best_file = f'{new_Folder}/{kernel_type}_Best_fold{fold}'
last_file = f'{new_Folder}/{kernel_type}_LastFile{fold}'
for epoch in range(start_epoch, n_epochs+start_epoch):
    print(time.ctime(), 'Epoch:', epoch)
    scheduler.step(epoch-1)

    train_loss = train_epoch(train_loader, optimizer)
    val_loss, acc, qwk = val_epoch(valid_loader)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {np.mean(train_loss):.5f}, val loss: {np.mean(val_loss):.5f}, acc: {(acc):.5f}, qwk: {(qwk):.5f}'
    print(content)
    with open(f'{new_Folder}/log_{kernel_type}.txt', 'a') as appender:
        appender.write(content + '\n')

    if qwk > qwk_max:
        print('score2 ({:.6f} --> {:.6f}).  Saving model ...'.format(qwk_max, qwk))
        filename = f'{best_file}_lastEpoch_{epoch}_kappa_{qwk}.pth'
        torch.save(model.state_dict(), filename)
        qwk_max = qwk
    else:
      filename = f'{last_file}_lastEpoch_{epoch}_kappa_{qwk}.pth'
      torch.save(model.state_dict(), filename)

torch.save(model.state_dict(), os.path.join(f'{new_Folder}/{kernel_type}_KAPPA:{qwk_max}_final_fold{fold}.pth'))





