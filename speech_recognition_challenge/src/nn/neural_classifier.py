import csv
import gzip
import json
import pickle
import sys
from collections import OrderedDict
import uuid
import os

import cv2
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import numpy as np

import config
import nn.losses as losses_utils
import nn.tools as tools
from skimage.io import imsave
import fegolib as fgl
import warnings
import zipfile
from sklearn.metrics import roc_auc_score, accuracy_score

sys.path.insert(0,'..')
warnings.filterwarnings("ignore")

class Classifier:
    def __init__(self, net_tuple, train_loader, valid_loader, valid_loader_oof_list,
                 test_loader_list, output_folder, fold_num, load_model_from_file = None):
        """
        """
        self.nn_name = net_tuple[0]
        self.net = net_tuple[1]()
        self.valid_loader = valid_loader
        self.train_loader = train_loader
        self.valid_loader_oof_list = valid_loader_oof_list
        self.test_loader_list = test_loader_list
        self.use_cuda = config.USE_CUDA
        self.threshold = 0.5
        self.load_model_from_file = load_model_from_file
        self.fold_num = fold_num
        self.output_folder = output_folder
        self.output_folder_predictions = self.output_folder + 'predictions/'

        self.lr_continue = 1e-3
        self.lr_min = 1e-8 #so if lr == lr_min - than early stop
        # self.optimizer =  optim.SGD(self.net.parameters(), lr = self.lr_continue, momentum = 0.9, weight_decay = 0.0001)
        # self.optimizer =  optim.RMSprop(self.net.parameters(), lr = self.lr_continue, momentum = 0.9, weight_decay = 0.0001)
        self.optimizer =  optim.Adam(self.net.parameters(), lr = self.lr_continue)

        if not os.path.exists(self.output_folder_predictions):
            os.makedirs(self.output_folder_predictions)
            
        self.sub_name = self.output_folder + self.nn_name + '_' + str(uuid.uuid4())

        if self.load_model_from_file:
            self.net.load_state_dict(torch.load(self.load_model_from_file))

    def _criterion(self, probs, labels):
        res = torch.nn.CrossEntropyLoss().forward(probs, labels)
        # res = torch.nn.BCELoss().forward(probs, labels)
        # res = losses_utils.StableBCELoss().forward(probs, labels)
        
        return res

    def _validate_epoch(self):
        losses = tools.AverageMeter()
        accuracies = tools.AverageMeter()

        it_count = len(self.valid_loader)
        batch_size = self.train_loader.batch_size
        with tqdm(total = it_count, desc = "Validating", leave = False) as pbar:
            for ind, loader_dict in enumerate(self.valid_loader):
                #train
                loss, acc = self._batch_train_validation(loader_dict, volatile = True)
                
                losses.update(loss.data[0], batch_size)
                accuracies.update(acc, batch_size)
                pbar.update(1)

        return losses.avg, accuracies.avg

    def _train_epoch(self, epoch_id, epochs):
        losses = tools.AverageMeter()
        accuracies = tools.AverageMeter()

        # Total training files count / batch_size
        batch_size = self.train_loader.batch_size
        it_count = len(self.train_loader)

        with tqdm(total = it_count,
                  desc = "Epochs {}/{}".format(epoch_id + 1, epochs),
                  bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{remaining}{postfix}]'
                  ) as pbar:

            for ind, loader_dict in enumerate(self.train_loader):
                
                #train
                loss, acc = self._batch_train_validation(loader_dict, volatile = False)

                # backward + optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses.update(loss.data[0], batch_size)
                accuracies.update(acc, batch_size)

                # Update pbar
                pbar.set_postfix(OrderedDict(loss='{0:1.5f}'.format(loss.data[0]), acc='{0:1.5f}'.format(acc)))
                pbar.update(1)
        return losses.avg, accuracies.avg

    def _batch_train_validation(self, loader_dict, volatile):
        #volatile = False for train
        images = loader_dict['img']
        target = loader_dict['target'].type(torch.LongTensor)

        if self.use_cuda:
            images = images.cuda()
            target = target.cuda()
            
        images = Variable(images, volatile = volatile)
        target = Variable(target, volatile = volatile)

        # forward
        probs = self.net.forward(images)

        loss = self._criterion(probs, target)
        acc = accuracy_score(target.data.cpu().numpy(), nn.Softmax()(probs).data.cpu().numpy().round().argmax(axis=1))
        
        return loss, acc
    
    def train(self, epochs, threshold = 0.5):
        """
        """
        self.weights_folder = self.output_folder + 'fold_w_%d/'%self.fold_num
        self.weights_folder = fgl.utils.path_that_not_exist(self.weights_folder, create = True)

        if self.use_cuda:
            self.net.cuda()

        print("Training on {} samples and validating on {} samples "
              .format(len(self.train_loader.dataset), len(self.valid_loader.dataset)))

         #continue train and write weights to existing folder
        training_info_from_file_flag = False
        
        start_epoch_continue = 0
        print('Training from scratch')

        if not training_info_from_file_flag:
            train_loss_list = []
            valid_loss_list = []
            train_acc_list =[]
            valid_acc_list =[]
            lr_list = []
            weights_list =[]

        lr_scheduler = ReduceLROnPlateau(self.optimizer, 'min', 
                                    patience = 4,  verbose = True, 
                                    min_lr = self.lr_min
                                    )

        for epoch_id in range(start_epoch_continue, epochs):
            #early stopping
            if (epoch_id > 0): 
                if (current_lr <= 2 * self.lr_min) :
                    print('Early stopping')
                    break

            self.net.train()
            # Run a train pass on the current epoch
            train_loss, train_acc = self._train_epoch(epoch_id, epochs)

            # switch to evaluate mode
            self.net.eval()
            valid_loss, valid_acc = self._validate_epoch()
            print("train_loss = {:03f}, val_loss = {:03f}, train_acc = {:03f}, val_acc = {:03f} -- {} {}" \
                                .format(train_loss,valid_loss, train_acc, valid_acc, self.nn_name, self.fold_num))
            print("")

            #get current lr
            current_lr = self.optimizer.param_groups[0]['lr']
            
            #save weights on each epoch    
            weights_final_name = self.weights_folder + 'w_' + str(epoch_id) + '.dat'
            weights_final_name = fgl.utils.path_that_not_exist(weights_final_name)
            torch.save(self.net.state_dict(), weights_final_name)

            #saving to pandas df
            train_loss_list.append(train_loss)
            valid_loss_list.append(valid_loss)
            train_acc_list.append(train_acc)
            valid_acc_list.append(valid_acc)
            lr_list.append(current_lr)
            weights_list.append(str(epoch_id))

            #update loss info on every step
            self.training_log_info = pd.DataFrame({'weight' : weights_list, 'train_loss' : train_loss_list, 'valid_loss' : valid_loss_list,
                                                'train_acc' : train_acc_list, 'valid_acc' : valid_acc_list, 'lr' : lr_list })
            self.training_log_info.sort_values('valid_loss', inplace = True)
            self.training_log_info.to_csv(self.output_folder + "trainig_log%d.csv"%self.fold_num, index = False)

            #lr step
            lr_scheduler.step(valid_loss, epoch_id)
    
            #if we have decreased lr - load training from the best previous weight for this lr
            if  self.optimizer.param_groups[0]['lr']  < current_lr:
                best_weight_name = self.training_log_info[self.training_log_info['lr'] == current_lr].sort_values('valid_loss').head(1)['weight'].item()
                print('Decrease lr - loading best weight %s for lr %f'%(str(best_weight_name), current_lr))
                self.net.load_state_dict(torch.load( self.weights_folder + 'w_' + str(best_weight_name) + '.dat' ))

    def predict(self):
        #if training from scratch - then load best weight
        if not self.load_model_from_file:
            best_weight_name = self.training_log_info.sort_values('valid_loss').head(1)['weight'].item()
            print("Loading best weight for prediction %s"%(str(best_weight_name)))
            self.net.load_state_dict(torch.load( self.weights_folder + 'w_' + str(best_weight_name) + '.dat' ))
            
        # Switch to evaluation mode
        if self.use_cuda:
            self.net.cuda()
        self.net.eval()

        oof_prediction, new_cols = self.predict_single_loader('val')
        test_pred_sub, new_cols = self.predict_single_loader('test')
        return oof_prediction,  test_pred_sub, new_cols

    def predict_single_loader(self, mode):
        """choice - val or test"""
        if mode == 'val': 
            loader_list = self.valid_loader_oof_list
        elif mode == 'test':
            loader_list = self.test_loader_list

        new_cols = []
        for loader_num, loader in enumerate(loader_list):
            print('Predicting fold %d on loader %s : %d / %d'%(self.fold_num, mode, loader_num+1, len(loader_list)))
            it_count = len(loader)
            predictions = []
            imgs_names = []

            with tqdm(total = it_count, desc = "Predicting") as pbar:
                for ind, loader_dict in enumerate(loader):
                    
                    images = loader_dict['img']
                    img_id = loader_dict['id']

                    if self.use_cuda:
                        images = images.cuda()

                    images = Variable(images, volatile = True)

                    # forward
                    # print('predicting ', images.shape)
                    probs = self.net(images)
                    probs = nn.Softmax()(probs)

                    # Save the predictions
                    for (pred, name) in zip(probs, img_id):
                        pred_arr = pred.data.cpu().numpy()

                        imgs_names.append(name)
                        predictions.append(pred_arr)

                    pbar.update(1)

            current_cols = []
            for target_num in np.arange(len(predictions[0])):
                col = config.pred_col_name_template.substitute(aug_num = loader_num, target_num = target_num)
                current_cols.append(col)

            new_cols += current_cols

            s = pd.DataFrame(np.array(predictions))
            s.columns =  current_cols  
            s['id'] = imgs_names

            if loader_num > 0:
                sub = sub.merge(s, on = 'id') 
            else:
                sub = s
        
        if mode == 'test':
            # assert len(sub) == config.LEN_DFS['test']
            sub.to_csv(self.output_folder_predictions + 'test_prediction_fold_%d.csv'%(self.fold_num), index = False)
        elif mode == 'val':
            sub.to_csv(self.output_folder_predictions + 'oof_prediction_fold_%d.csv'%(self.fold_num), index = False)

        return sub, new_cols
