import matplotlib
matplotlib.use('Agg')

import config
import gc
import glob
import os
import uuid
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import time
import random

import fegolib as fgl
import img.augmentation as aug
import nn.classifier
import torch
from utils import utils as utils
import shutil

from dataset.loader import Dataset, ImageDataset, CustomPredictionDataset
from dataset.samplers import SilenceBinaryRandomSampler, UnknownsRandomSampler

# 31 class nets
from nets.test_net import TestNet
from nets.incep_res_v2 import IncepResV2
from nets.fb_resnet import FBResnet152
from nets.dense_net import DenseNet, DenseNet_121
from nets.dpn import DPN92
from nets.resnext import Resnext101
from nets.vgg import VGG19bn
from nets.resnets import resnet18, resnet34, resnet50
from nets.squeezenet import SqueezeNet
from nets.drn import DRN107

# aux
from nets.vgg_aux import VGG19bn_aux

# binary nets
from nets_binary.vgg import binary_VGG19bn

# 1 d net
from nets_1d.testnet import TestNet_one_d

from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
import torch
import json
import time

batch_size = 64
epochs = 100

mode = '31class' # '31class' 'binary'
include_double_words = True
#--------------------------------
pseudo_file = None # 'nn_8/pseudo_nn_8_percent_20.csv' # None
RS_list = [
            15 * 1, 
            # 15 * 2,
            # 15 * 10,
            # 15 * 20,
            ]
#-------------------------------

# time.sleep(60 * 60 * 2 *3) #sleep 2 hours

nets_list = [
            # TestNet,
            # IncepResV2,
            # FBResnet152,
            # DenseNet,
            DPN92,
            # Resnext101,
            # VGG19bn,
            # resnet18, 
            # resnet34, 
            # resnet50,
            # VGG19bn_aux
            # SqueezeNet,
            # DenseNet_121,
            # DRN107,
            # TestNet_one_d
]

class Main(object):
    def __init__(self, net_tuple, RS, folder_for_model_weights = None, predict_custom_flag = False):
        """
        folder_for_model_weights - folder where the weight for model are stored
        """
        self.RS = RS
        print('Random seed = %d'%RS)

        #random seed
         #--------------------------
        np.random.seed(RS)
        random.seed(RS)

        torch.cuda.manual_seed_all(RS)
        torch.manual_seed(RS)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(RS)
            torch.cuda.manual_seed(RS)
        #--------------------------

        self.net_name = net_tuple[0]
        self.train_flag = True
        self.folder_for_model_weights = folder_for_model_weights
        if self.folder_for_model_weights is not None:
            self.train_flag = False

        self.num_folds = 5
        self.best_weights = []
        self.training_logs = []
        self.pseudo_file = pseudo_file
        self.augs_func_list = [None, aug.random_speed, aug.random_noise, aug.random_shift] 
        self.mode = mode
        self.predict_custom_flag = predict_custom_flag

        if self.folder_for_model_weights:
            self.train_flag = False
            print('No train.')
            self.get_weights_name()

        self.epochs = epochs

        
        if self.mode == '31class':
            name = 'nn/'
            self.silence_binary_flag = False
            if self.pseudo_file:
                name = 'pseudo_n_n/'
            elif (self.folder_for_model_weights is not None) & (self.predict_custom_flag):
                name = 'custom_predictions/'
                self.custom_dataset = CustomPredictionDataset('silence')
        if "one_d" in self.net_name:
            name = 'one_d/'


        elif self.mode == 'binary':
            self.silence_binary_flag = True
            name = 'binary/'

        if self.net_name  == 'TestNet':
            self.epochs = 1
            name = '_nn_test_script/%s/'%name      

        #folders
        self.folder_name = config.OUTPUT_FOLDER + name
        self.folder_name = fgl.utils.path_that_not_exist(self.folder_name, create = True)
        utils.save_src_to_zip(self.folder_name, ['data'])
        
        if (self.folder_for_model_weights is not None) & (self.predict_custom_flag):
            shutil.copy(config.OUTPUT_FOLDER + self.folder_for_model_weights + '/label_encoder.dump',self.folder_name )


        self.splits_folder = self.folder_name + 'splits/'
        if not os.path.exists(self.splits_folder):
            os.makedirs(self.splits_folder)

        self.sub_name = self.folder_name  + self.net_name + '_' + str(uuid.uuid4()) + '.csv'

        self.test_prediction = None
        self.train_prediction = None
        print('Using %s net'%net_tuple[0])
        print('Saving to %s'%self.folder_name)

       # dump settings
        settings_dict = {}
        settings_dict['RS'] = self.RS
        settings_dict['arch'] = self.net_name
        settings_dict['test_augs_list'] = [x.__name__ if x is not None else x for x in self.augs_func_list ]
        with open(self.folder_name + 'settings.json', 'w') as f:
            json.dump(settings_dict, f)

    def __call__(self):
        self.ds = Dataset(RS = self.RS, proj_folder = self.folder_name, 
                            pseudo_file = self.pseudo_file,
                            silence_binary = self.silence_binary_flag,
                            include_double_words = include_double_words
                            )

        test_dataset = self.ds.test       
        if self.predict_custom_flag:
            test_dataset = self.custom_dataset.test 
        test_loader_list = self.generate_tta_loader_list(test_dataset)

        for fold_num in [4,3,2,1,0]:#np.arange(self.num_folds):
            #train
            train_ids = self.ds.train_ids_list[fold_num]
            outfile = open(self.splits_folder + 'train_%d.txt'%fold_num, 'w')
            for item in train_ids:  outfile.write("%s\n" % str(item))

            valid_ids = self.ds.val_ids_list[fold_num]
            outfile = open(self.splits_folder + 'val_%d.txt'%fold_num, 'w')
            for item in valid_ids:  outfile.write("%s\n" % str(item))

            print('Train index', train_ids[:15])
            if mode == '31class':
                if not include_double_words:
                    train_ds = ImageDataset(self.ds.train[self.ds.train[config.id_col].isin(train_ids)],
                                        include_target = True,
                                        X_transform = aug.data_transformer)
                    val_ds = ImageDataset(self.ds.train[self.ds.train[config.id_col].isin(valid_ids)], 
                                        include_target = True,
                                        X_transform = None)

                    train_loader = DataLoader(train_ds, batch_size,
                                        sampler = RandomSampler(train_ds),
                                        num_workers = config.THREADS,
                                        pin_memory= config.USE_CUDA )
                    valid_loader = DataLoader(val_ds, batch_size,
                                        num_workers = config.THREADS,
                                        pin_memory = config.USE_CUDA )
                else:
                    train_ds = ImageDataset(self.ds.train,
                                    include_target = True,
                                    X_transform = aug.data_transformer)
                    val_ds = ImageDataset(self.ds.train, 
                                        include_target = True,
                                        X_transform = None)
                    train_loader = DataLoader(train_ds, batch_size,
                                        sampler = UnknownsRandomSampler(self.ds.train[self.ds.train[config.id_col].isin(train_ids)]),
                                        num_workers = config.THREADS,
                                        pin_memory = config.USE_CUDA )              
                    valid_loader = DataLoader(val_ds, batch_size,
                                        sampler = UnknownsRandomSampler(self.ds.train[self.ds.train[config.id_col].isin(valid_ids)]),
                                        num_workers = config.THREADS,
                                        pin_memory = config.USE_CUDA )                     

            elif mode == 'binary':
                train_ds = ImageDataset(self.ds.train,
                                    include_target = True,
                                    X_transform = aug.data_transformer)
                val_ds = ImageDataset(self.ds.train, 
                                    include_target = True,
                                    X_transform = None)


                train_loader = DataLoader(train_ds, batch_size,
                                    sampler = SilenceBinaryRandomSampler(self.ds.train[self.ds.train[config.id_col].isin(train_ids)]),
                                    num_workers = config.THREADS,
                                    pin_memory = config.USE_CUDA )              
                valid_loader = DataLoader(val_ds, batch_size,
                                    sampler = SilenceBinaryRandomSampler(self.ds.train[self.ds.train[config.id_col].isin(valid_ids)]),
                                    num_workers = config.THREADS,
                                    pin_memory = config.USE_CUDA )
            valid_loader_oof_list = self.generate_tta_loader_list(self.ds.train[self.ds.train[config.id_col].isin(valid_ids)])

            #train
            if self.train_flag :
                classifier = nn.classifier.Classifier(net_tuple = net_tuple, train_loader = train_loader, 
                                                    valid_loader_oof_list = valid_loader_oof_list,
                                                    valid_loader = valid_loader, test_loader_list = test_loader_list, 
                                                    output_folder = self.folder_name, fold_num = fold_num,
                                                    load_model_from_file = None, mode = self.mode)
                classifier.train(self.epochs)
            else: #load model and predict 
                classifier = nn.classifier.Classifier(net_tuple = net_tuple, train_loader = train_loader, 
                                                    valid_loader_oof_list = valid_loader_oof_list,
                                                    valid_loader = valid_loader, test_loader_list = test_loader_list, 
                                                    output_folder = self.folder_name, fold_num = fold_num,
                                                    load_model_from_file = self.model_weights_dict[fold_num],
                                                    mode = self.mode)
            
            #predict
            oof_train, test_pred_sub, self.aug_col_list = classifier.predict()

            if self.train_flag :
                training_log_info = classifier.training_log_info
                self.best_weights.append(training_log_info.head(1)['weight'].item())
                self.training_logs.append(training_log_info)

            del classifier; gc.collect

            #concat oof for train
            if isinstance(self.train_prediction, pd.DataFrame):
                self.train_prediction = pd.concat([self.train_prediction , oof_train])
            else:
                self.train_prediction = oof_train

            #merge test predictions
            if isinstance(self.test_prediction, pd.DataFrame):
                self.test_prediction = self.test_prediction.merge(test_pred_sub, on = 'id')
            else:
                self.test_prediction = test_pred_sub


        #calculate score across folds
        self.errors_dict = {}
        loss_list = np.array([x.head(1)['valid_loss'].item() for x in self.training_logs])

        self.errors_dict['val_std'] = np.std(loss_list)
        self.errors_dict['val_mean'] = np.mean(loss_list)
        self.errors_dict['best_dict'] = self.best_weights
                
        with open(self.folder_name + 'results.json', 'w') as fp:
            json.dump(self.errors_dict, fp)

        #calculating mean score on train
        # train_prediction_for_error = self.train_prediction.copy()
        # train_prediction_for_error.rename(columns = {config.target_col : config.target_col + '_pr'}, inplace = True)
        # self.ds.train.rename(columns = {config.target_col : config.target_col + '_gt'}, inplace = True)
        # df = self.ds.train.merge(train_prediction_for_error, on = 'id')
        # df['error'] = np.abs(df[config.target_col + '_gt'] - df[config.target_col + '_pr'])
        # self.errors = df['error'].values


    def plot(self):
        #fold performance
        f, ax = plt.subplots(1, self.num_folds, figsize = (20, 6))
        for i in np.arange(self.num_folds):
            self.training_logs[i].sort_values('weight', inplace = True)
            ax[i].plot(self.training_logs[i]['weight'].tolist(), self.training_logs[i]['train_loss'].tolist(), color = 'blue', label = 'train')
            ax[i].plot(self.training_logs[i]['weight'].tolist(), self.training_logs[i]['valid_loss'].tolist(), color = 'red', label = 'val')
            ax[i].set_title('Fold %d'%i)
            ax[i].legend(loc = 'upper right', shadow = True)
            ax[i].grid()
            ax[i].set_ylim([0, 0.5])
        
        plt.savefig(self.folder_name + 'img_loss.png', dpi = 100)

        #train test prediction distribution and train error
        # f, ax = plt.subplots(2, 1, figsize = (20, 15))
        # ax[0].hist(self.errors, normed = True,  bins = 200)
        # ax[0].set_title('train error')
        # ax[0].grid()

        # ax[1].hist(self.test_prediction[config.target_col].tolist(), normed = True, label = 'test', bins = 100)
        # ax[1].hist(self.train_prediction[config.target_col].tolist(), normed = True, label = 'train', bins = 100)
        # ax[1].legend(loc = 'upper right')
        # ax[1].set_title('Train / test prediction distibution')
        # ax[1].grid()

        # plt.savefig(self.folder_name + 'img_distibutions.png', dpi = 100)

    def get_weights_name(self):
        self.model_weights_dict = {}
        self.parse_folds(config.OUTPUT_FOLDER + '/' + self.folder_for_model_weights + '/')

    def parse_folds(self, folder):
        csv_files = glob.glob(folder +  '*.csv')
        training_log_file_names = [x for x in csv_files if 'trainig_log' in x]
        print(training_log_file_names)
        print('Found %d folds'%self.num_folds)
        for i in np.arange(self.num_folds):
            training_info = pd.read_csv(training_log_file_names[i])
            training_info.sort_values('valid_loss', inplace = True)
            best_weight_name = 'w_' + str(training_info.head(1)['weight'].item()) + '.dat'
            self.model_weights_dict[i] = folder + '/fold_w_%d/'%i + best_weight_name

            self.training_logs.append(training_info)
            self.best_weights.append(training_info.head(1)['weight'].item())

    def remove_unnecessary_weights(self): 
        print('Removing unnecessary weights.')
        folder = self.folder_name
        csv_files = glob.glob(folder +  '*.csv')
        training_log_file_names = [x for x in csv_files if 'trainig_log' in x]
        print('Found %d folds'%self.num_folds)
        for i in np.arange(self.num_folds):
            training_info = pd.read_csv(training_log_file_names[i])
            training_info.sort_values('valid_loss', inplace = True)
            #remove_unnecessary_weights(self):
            top_3_weights = training_info.head(2)['weight'].tolist()
            unnecessary_weigths =  training_info[~training_info['weight'].isin(top_3_weights)]['weight'].tolist()
            for bad_weight in unnecessary_weigths:
                os.remove(folder + '/fold_w_%d/'%i + 'w_' + str(bad_weight) + '.dat')

    def generate_tta_loader_list(self, df):
        res_list = []

        print('predicting on %d augmentations'%len(self.augs_func_list))

        for augs_func in self.augs_func_list:
            ds = ImageDataset(df,  include_target = False,  X_transform =  augs_func, u = 666)

            tl = DataLoader(ds, batch_size,
                    num_workers = config.THREADS,
                    pin_memory= config.USE_CUDA,
                    )

            res_list.append(tl) 

        return res_list

if __name__ == "__main__":
    for i in enumerate(np.arange(len(nets_list))):
        i = i[0]
        nets_list[i] = [nets_list[i].__name__, nets_list[i]]
        print(nets_list[0][0])
        for RS in RS_list:
            for net_tuple in nets_list:
                m = Main(net_tuple, RS, folder_for_model_weights = None)
                m()
                m.plot()

                del m; gc.collect()
                
                print('Waiting delay')
                time.sleep(60) 



# if __name__ == "__main__":
#     folders_list = ["nn_11"]
#     for folder in folders_list:
#         print("predicting %s"%folder)
#         data_folder = config.OUTPUT_FOLDER + folder +'/'
#         arch_name = json.load(open( data_folder + 'settings.json', 'r'))['arch']
#         arch = eval(arch_name)
#         # loading net name

#         net_tuple = (arch_name, arch)
#         m = Main(net_tuple, RS = RS_list[0], folder_for_model_weights = folder, predict_custom_flag = True)
#         m()
