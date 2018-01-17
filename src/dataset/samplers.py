import torch.utils.data as data
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data.sampler import RandomSampler, SequentialSampler, Sampler
from sklearn import preprocessing
from torch.utils.data import DataLoader
from sklearn.utils import shuffle
import numpy as np
import os
import cv2
import pandas as pd
import sys
from tqdm import tqdm
import torch
import glob
from skimage.io import imread
import skimage.transform
from PIL import Image
import cv2
import joblib
import random

from scipy.io import wavfile
from scipy import signal
import loader

sys.path.insert(0,'..')
import config
from img import transformer
import img.augmentation as aug


class SilenceBinaryRandomSampler(Sampler):
    def __init__(self, data, silence_probability = 0.5):
        self.data = data
        self.silence_probability = silence_probability
        self.speech_probability   = 1 - self.silence_probability

        self.silence_num = len(self.data[self.data['target'] == 1])
        self.speech_num = int((self.silence_num / self.silence_probability) * self.speech_probability)
        print('Silence num = %d, speech num = %d'%(self.silence_num, self.speech_num))

        self.length = self.silence_num + self.speech_num

    def __iter__(self):
       
        l  = []
        if self.silence_num>0:# silence
            silence_list = self.data[self.data['target'] == 1].index.tolist()
            random.shuffle(silence_list)
            l += silence_list

        if self.speech_num>0:# speech
            speech_list = self.data[self.data['target'] != 1].index.tolist()
            random.shuffle(speech_list)
            speech_list = speech_list[:self.speech_num]
            l +=  speech_list
        
        assert(len(l) == self.length), "%d, %d"%(len(l), self.length)
        random.shuffle(l)
        return iter(l)

    def __len__(self):
        return self.length


class UnknownsRandomSampler(Sampler):
    def __init__(self, data, unknown_probability = 1 / float(31)):
        self.data = data
        self.unknown_probability = unknown_probability
        self.speech_probability   = 1 - self.unknown_probability

        self.speech_num = len(self.data[self.data['target'] != 31])
        self.unknown_num = int(self.speech_num * self.unknown_probability)
        print(self.unknown_probability, (self.speech_num) * self.unknown_probability)
        print('Unkown num = %d, speech num = %d'%(self.unknown_num, self.speech_num))

        self.length = self.unknown_num + self.speech_num

    def __iter__(self):
       
        l  = []
        # unknown
        unknown_list = self.data[self.data['target'] == 31].index.tolist()
        random.shuffle(unknown_list)
        unknown_list = unknown_list[:self.unknown_num]
        l += unknown_list

        # speech
        speech_list = self.data[self.data['target'] != 31].index.tolist()
        random.shuffle(speech_list)
        l +=  speech_list
        
        assert(len(l) == self.length), "%d, %d"%(len(l), self.length)
        random.shuffle(l)
        return iter(l)

    def __len__(self):
        return self.length

# def collate(batch):
#     batch_size = len(batch)
#     num = len(batch[0])
#     indices = [batch[b][num-1]for b in range(batch_size)]
#     tensors = torch.stack([batch[b][0]for b in range(batch_size)], 0)
#     if batch[0][1] is None:
#         labels = None
#     else:
#         labels = torch.from_numpy(np.array([batch[b][1]for b in range(batch_size)])).long()
#     return [tensors,labels,indices]


if __name__ == '__main__':
    ds = loader.Dataset(RS = 15, proj_folder = config.DATA_FOLDER + 'zzz/', include_double_words = True)


    batch_size = 100
    train_ds = loader.ImageDataset(ds.train, include_target = True,
                             X_transform = aug.data_transformer
                            )
    train_loader = DataLoader(train_ds, batch_size,
                            sampler = UnknownsRandomSampler(ds.train),
                            num_workers = 5,
                            pin_memory= config.USE_CUDA )

    for i, dict_ in enumerate(train_loader):
        for j in np.arange(batch_size):
            print(dict_['img'].min(), dict_['img'].max(), dict_['img'].shape)
            print(dict_['target'])

        if i > 1:
            break