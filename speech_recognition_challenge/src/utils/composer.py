import pandas as pd
import uuid
import numpy as np
import fegolib as fgl
from sklearn.metrics import roc_auc_score, log_loss
import json
import os
import glob

import config
import utils

EXCLUDE_FOLDERS = ['nn_11', 'nn_12', 'nn_16', 'nn_23'] + ['nn_%d'%x for x in np.arange(9)] + ['nn'] 
NET_NAMES = ['SENet2048']

#-------------------------
RS_list = [
            15 * 1, 
            15 * 2,
            15 * 10,
            15 * 20,
            ]
#-------------------------


class Composer(object):
    """
    select the best prediction for each fold among all the same networks 
    with the same folds and compose best prediction from the best weights
    """
    def __init__(self, RS, arch):
        self.EXCLUDE_FOLDERS = EXCLUDE_FOLDERS
        self.RS = RS
        self.n_splits = 5
        self.arch = arch
        self.save_folder = fgl.utils.path_that_not_exist(config.OUTPUT_FOLDER + 'cmpsr/', create = True)
        self.train_pred = None
        self.test_pred = None

        utils.save_src_to_zip(self.save_folder, ['data'])
        with open(self.save_folder + 'RS.txt', 'w') as f:
            f.write('%d' % self.RS)
        with open(self.save_folder + 'arch.txt', 'w') as f:
            f.write('%s' % self.arch)

        self.nn_folders = [ x for x in glob.glob(config.OUTPUT_FOLDER + '*/') if\
                                            ('nn' in x.split('/')[-2]) and  \
                                            ('_nn_test_script' not in x.split('/')[-2]) and \
                                            (x.split('/')[-2] not in self.EXCLUDE_FOLDERS)]

        self.valid_folders = self.find_corresponding_folders()
        print('Found %d valid folders'%len(self.valid_folders))
        self.read_logs()
        self.make_composit()
    
    def make_composit(self):
        self.best_log['full_path_oof'] = self.best_log['folder'] + 'predictions/' + 'oof_prediction_fold_' + self.best_log['fold'].astype(str) + '.csv'
        self.best_log['full_path_test'] = self.best_log['folder'] + 'predictions/' + 'test_prediction_fold_' + self.best_log['fold'].astype(str) + '.csv'
        self.best_log.to_csv(self.save_folder + 'best_log.csv', index = False)

        for fold in np.arange(self.n_splits):
            single_dict = self.best_log[self.best_log['fold'] == fold].head(1).to_dict(orient = 'list')
            train_oof = pd.read_csv(single_dict['full_path_oof'][0])
            test = pd.read_csv(single_dict['full_path_test'][0])

            aug_cols = [x for x in train_oof.columns.tolist() if 'target_aug' in x]
            train_oof[config.target_col] = train_oof[aug_cols].mean(axis = 1)
            for c in aug_cols:
                test.rename(columns = {c : c + '_fold%d'%fold}, inplace = True)

            if isinstance(self.train_pred, pd.DataFrame):
                self.train_pred = pd.concat([self.train_pred, train_oof])
                self.test_pred = self.test_pred.merge(test, on = 'id')
            else:
                self.train_pred = train_oof
                self.test_pred =  test
        
        #find test mean
        for c in aug_cols:
            self.test_pred[c] = self.test_pred[[c + '_fold%d'%fold for fold in np.arange(self.n_splits) ]].mean(axis = 1)
        self.test_pred[config.target_col] = self.test_pred[aug_cols].mean(axis = 1)

        assert len(self.train_pred) == config.LEN_DFS['train']
        assert self.train_pred['id'].nunique() == config.LEN_DFS['train']
        assert len(self.test_pred) == config.LEN_DFS['test']

        self.train_pred.to_csv(self.save_folder + 'train_pred_oof.csv', index = False)
        self.test_pred[['id'] + aug_cols + [config.target_col]].to_csv(self.save_folder + 'test_pred.csv', index = False)

    def read_logs(self):
        logs = []
        for fold in np.arange(self.n_splits):
            for folder in self.valid_folders:
                single_log_best = pd.read_csv(folder + 'trainig_log%d.csv'%fold, usecols = ['valid_loss'])['valid_loss'].min()
                logs.append({'fold' : fold, 'folder' : folder, 'valid_loss' : single_log_best})

        logs = pd.DataFrame(logs)

        #saving best weights for fold
        for fold in np.arange(self.n_splits):
            single_log = logs[logs['fold'] == fold].copy().sort_values('valid_loss', ascending = True)
            single_log.drop('fold', 1).to_csv(self.save_folder + 'best_log_fold_%d.csv'%fold, index = False)

        #saving only best weight
        self.best_log = logs.sort_values(['fold', 'valid_loss'], ascending = True).drop_duplicates('fold', keep = 'first')
        self.best_log.to_csv(self.save_folder + 'best_log.csv', index = False)

        #find score improvement
        best_log_stats = {'mean' : self.best_log['valid_loss'].mean(), 'std': self.best_log['valid_loss'].std()}
        print('BSM',{'mean' : logs.groupby('folder')['valid_loss'].mean().to_frame().reset_index().head(1).to_dict(),
                    'std' : logs.groupby('folder')['valid_loss'].std().to_frame().reset_index().head(1).to_dict() })
        print('Composer scsore' , best_log_stats)
        with open(self.save_folder + 'results.json', 'w') as fp:
            json.dump(best_log_stats, fp)

    def find_corresponding_folders(self):
        res = []
        for folder in self.nn_folders:
            if (self.check_folder_RS(folder)) and (utils.check_folder_arch(self.arch, folder)):
                res.append(folder)
                print('Valid folder', folder)

        with open(self.save_folder + 'valid_folders.json', 'w') as fp:
            json.dump({'valid_folders' : res}, fp)

        return res

    def check_folder_RS(self, folder_path):
        with open(folder_path + 'RS.txt') as f:
            RS_txt = f.readlines()  
        RS_txt = int(RS_txt[0]) 

        res = False
        if RS_txt == self.RS:
            res = True

        return res
        

if __name__ == '__main__':
    for RS in RS_list:
        for arch in NET_NAMES:                                       
            Composer(RS, arch)        
