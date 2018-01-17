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
from nn.neural_classifier import Classifier
import torch
from utils import utils as utils

from dataset.neural_classifier_loader import Dataset, ImageDataset
from nets_clf.fc_net import FC_net

from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
import torch
import json
import time

batch_size = 1024
epochs = 1000

list_of_predictions_folders =  ['nn_0', 'nn_1', 'nn_3', 'nn_4', 'nn_5', 'nn_6', 'nn_7', 'nn_8', 'nn_9', 'nn_11']

#--------------------------------
pseudo_file = None#'test_confident_stckr_53_400.json'
RS_list = [
            15 * 1, 
            # 15 * 2,
            # 15 * 10,
            # 15 * 20,
            ]
#-------------------------------

# time.sleep(60 * 60 * 2) #sleep 2 hours

nets_list = [
            FC_net,
]

for i in enumerate(np.arange(len(nets_list))):
    i = i[0]
    nets_list[i] = [nets_list[i].__name__, nets_list[i]]
print(nets_list[0][0])

class Main(object):
    def __init__(self, net_tuple, RS, folder_for_model_weights = None):
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
        self.num_folds = 5
        self.best_weights = []
        self.training_logs = []
        self.pseudo_file = pseudo_file
        self.augs_func_list = [None] 
        self.list_of_predictions_folders = list_of_predictions_folders

        if self.folder_for_model_weights:
            self.train_flag = False
            print('No train.')
            self.get_weights_name()

        self.epochs = epochs

        name = 'neural_clf/'
        if self.net_name  == 'TestNet':
            self.epochs = 1
            name = '_nn_test_script/neural_clf/'
        elif self.pseudo_file:
            name = 'pseudo_neural_clf/'

        #folders
        self.folder_name = config.OUTPUT_FOLDER + name
        self.folder_name = fgl.utils.path_that_not_exist(self.folder_name, create = True)
        utils.save_src_to_zip(self.folder_name, ['data'])

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
        self.ds = Dataset(RS = self.RS, list_of_predictions_folders = list_of_predictions_folders, pseudo_file = self.pseudo_file)
        test_loader_list = self.generate_tta_loader_list(self.ds.test)

        for fold_num in [2,0,1,3,4]:#np.arange(self.num_folds):
            #train
            train_ids = self.ds.train_ids_list[fold_num]
            outfile = open(self.splits_folder + 'train_%d.txt'%fold_num, 'w')
            for item in train_ids:  outfile.write("%s\n" % str(item))

            print('Train index', train_ids[:15])
            train_ds = ImageDataset(self.ds.train[self.ds.train[config.id_col].isin(train_ids)],
                                    include_target = True,
                                    X_transform = aug.data_transformer)
            train_loader = DataLoader(train_ds, batch_size,
                                    sampler = RandomSampler(train_ds),
                                    num_workers = config.THREADS,
                                    pin_memory= config.USE_CUDA )
            #valid
            valid_ids = self.ds.val_ids_list[fold_num]
            outfile = open(self.splits_folder + 'val_%d.txt'%fold_num, 'w')
            for item in valid_ids:  outfile.write("%s\n" % str(item))

            val_ds = ImageDataset(self.ds.train[self.ds.train[config.id_col].isin(valid_ids)], 
                                    include_target = True,
                                    X_transform = None)
            valid_loader = DataLoader(val_ds, batch_size,
                                    num_workers = config.THREADS,
                                    pin_memory= config.USE_CUDA )
            valid_loader_oof_list = self.generate_tta_loader_list(self.ds.train[self.ds.train[config.id_col].isin(valid_ids)])

            #train
            if self.train_flag :
                classifier = Classifier(net_tuple = net_tuple, train_loader = train_loader, 
                                                    valid_loader_oof_list = valid_loader_oof_list,
                                                    valid_loader = valid_loader, test_loader_list = test_loader_list, 
                                                    output_folder = self.folder_name, fold_num = fold_num,
                                                    load_model_from_file = None)
                classifier.train(self.epochs)
            else: #load model and predict 
                classifier = Classifier(net_tuple = net_tuple, train_loader = train_loader, 
                                                    valid_loader_oof_list = valid_loader_oof_list,
                                                    valid_loader = valid_loader, test_loader_list = test_loader_list, 
                                                    output_folder = self.folder_name, fold_num = fold_num,
                                                    load_model_from_file = self.model_weights_dict[fold_num])
            
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

    def get_weights_name(self):
        self.model_weights_dict = {}
        self.parse_folds(config.OUTPUT_FOLDER + '/' + self.folder_for_model_weights + '/')

    def parse_folds(self, folder):
        csv_files = glob.glob(folder +  '*.csv')
        training_log_file_names = [x for x in csv_files if 'trainig_log' in x]
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
            top_3_weights = training_info.head(1)['weight'].tolist()
            unnecessary_weigths =  training_info[~training_info['weight'].isin(top_3_weights)]['weight'].tolist()
            for bad_weight in unnecessary_weigths:
                os.remove(folder + '/fold_w_%d/'%i + 'w_' + str(bad_weight) + '.dat')

    def generate_tta_loader_list(self, df):
        # print('DF len',df.shape)
        res_list = []

        # print(sorted(df.columns.tolist()))

        ds = ImageDataset(df,  include_target = False,   u = 666)
        tl = DataLoader(ds, batch_size,
                num_workers = config.THREADS,
                pin_memory= config.USE_CUDA,
                )

        res_list.append(tl)
        return res_list

if __name__ == "__main__":
    k = -1
    for RS in RS_list:
        for net_tuple in nets_list:
            k += 1
            if k > 0:
                print('Waiting delay')
                time.sleep(60) 

            m = Main(net_tuple, RS, folder_for_model_weights = None)
            m()
            m.plot()
            m.remove_unnecessary_weights()

            del m; gc.collect()
            

