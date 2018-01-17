import torch.utils.data as data
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data.sampler import RandomSampler, SequentialSampler
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
import gc

from scipy.io import wavfile
from scipy import signal
sys.path.insert(0,'..')
import config
from img import transformer
import img.augmentation as aug
from utils import utils
from utils import get_smarties

class Dataset:
    def __init__(self, RS, list_of_predictions_folders, n_folds = 5,pseudo_file = None):
        self.n_folds = n_folds
        self.train_ids_list = []
        self.val_ids_list = []
        self.RS = RS
        self.pseudo_file = pseudo_file
        self.list_of_predictions_folders = list_of_predictions_folders
        #
        print('Reading dataframes..')
        self.read_df()
        self.read_gt_train()
        self.train = self.train.merge(self.gt_train[[config.id_col, config.target_col]], on = config.id_col)
        assert len(self.train) == len(self.gt_train)

        assert self.train.shape[1] == len(self.list_of_predictions_folders) * 31 + 2, '%d , %d'%(self.train.shape[1],  len(self.list_of_predictions_folders) * 31 + 2)
        assert self.test.shape[1] == len(self.list_of_predictions_folders) * 31 + 1


        self.get_splits(self.list_of_predictions_folders[-1])
        # self.train.to_csv('train.csv', index = False)
        # self.test.to_csv('test.csv', index = False)

    def read_df(self):
        self.df = {}
        for train_test_flag in ['train', 'test']:
            self.df[train_test_flag] = None
            print('Reading %s'%train_test_flag)
            for folder_name in tqdm( self.list_of_predictions_folders):
                df_ = pd.read_csv(config.OUTPUT_FOLDER + folder_name + '/pred_%s.csv'%train_test_flag)
                df_.rename(columns = {str(i) : str(i) + '_' + folder_name for i in np.arange(31)}, inplace = True)
                if self.df[train_test_flag] is None:
                    self.df[train_test_flag] = df_
                else:
                    self.df[train_test_flag] = self.df[train_test_flag].merge(df_, on = config.id_col)
        
            assert len(self.df[train_test_flag]) == config.LEN_DFS[train_test_flag]
                
        self.train = self.df['train'].copy()
        self.test = self.df['test'].copy()

        del self.df; gc.collect

        # print(self.train.info())
        # print(self.test.info())

    def read_gt_train(self):
        self.gt_train = pd.read_csv(config.DATA_FOLDER + 'gt_train.csv')
        assert self.gt_train[config.target_col].nunique() == 31
        self.gt_train[config.target_col] = self.gt_train[config.target_col].map(config.mapping_dict)

    def get_splits(self, folder):
        self.train_ids_list = []
        for i in np.arange(5):
            self.train_ids_list.append(utils.read_list_from_file(config.OUTPUT_FOLDER + folder + '/splits/%s_%d.txt'%('train', i)))
        
        self.val_ids_list = []
        for i in np.arange(5):
            self.val_ids_list.append(utils.read_list_from_file(config.OUTPUT_FOLDER + folder + '/splits/%s_%d.txt'%('val', i)))

class ImageDataset(data.Dataset):
    def __init__(self, X_data, include_target, u = 0.5, X_transform = None):
        self.X_data = X_data
        self.include_target = include_target
        self.X_transform = X_transform
        self.u = u

        # divide data by columns
        self.id_data = self.X_data[config.id_col].values
        if self.include_target:
            self.target_data = self.X_data[config.target_col].values

        cols = [x for x in self.X_data.columns.tolist() if x not in [config.id_col, config.target_col]]
        self.X_data = self.X_data[cols].values
        # print('ImageDataset', self.X_data.shape)

    def __getitem__(self, index):
        img_id = self.id_data[index]
        img = self.X_data[index, :]
        img_numpy = img.astype(np.float32)
        img_torch = torch.from_numpy(img_numpy)

        dict_ = {'img' : img_torch,
                'id' : img_id
                }

        if self.include_target:
            dict_['target'] = self.target_data[index]
    
        # print('loader', img_torch.size(), self.include_target)

        return dict_

    def __len__(self):
        return len(self.X_data)

if __name__ == '__main__':
    ds = Dataset(15, list_of_predictions_folders = ['nn_0', 'nn_1', 'nn_3', 'nn_4', 'nn_5', 'nn_6'])
    print(ds.train.values.shape)
    print(ds.test.values.shape)

    batch_size = 2
    train_ds = ImageDataset(ds.test, include_target = False, u = 666)
                            
    train_loader = DataLoader(train_ds, batch_size,
                            num_workers = 5,
                            pin_memory= config.USE_CUDA )

    for i, dict_ in enumerate(train_loader):
        for j in np.arange(batch_size):
            print(dict_['img'].min(), dict_['img'].max(), dict_['img'].shape)
            # print(dict_['target'])

        if i > 1:
            break
