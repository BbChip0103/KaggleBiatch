import pandas as pd
import uuid
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
import json
import glob
import joblib
from tqdm import tqdm
import sys, os
sys.path.insert(0,'..')
import config
from utils import utils
from scipy.stats.mstats import gmean

#experiments
folder_names = { "nn/37": 1,
               }

class Blender(object):
    def __init__(self, folder_names = None, arch_name = None, mode = 'mean'):
        """
        2 methods:
            - use predifined folders - than pass folder_names
            - use architecture name to grab all folders with this arch
        2 modes: median and mean
        """
        self.arch = arch_name
        self.save_folder, _ = utils.create_path(os.path.join(config.OUTPUT_FOLDER, 'blnd', create = True)
        self.n_targets = 340
        self.folder_names = folder_names
        self.categories_mapping = json.load(open(os.path.join(str(), "kaggle_splits", "doodle", "category_mapping.json"), "r"))
        self.inverse_mapping_dict = {str(v):k for k,v in self.categories_mapping.items()}

        # self.mode =  mode
        # print('Selected mode : %s'%self.mode)
        #drop duplicates from dict
        result = {}
        for key, value in self.folder_names.items():
            if key not in result.keys():
                result[key] = value
        self.folder_names = result
        print(self.folder_names)

        self.process_data()

    def process_data(self):
        self.subs_dict = {}
        self.final_sub = {}
        RND = self.save_folder.split('/')[-1]

        for train_test_flag in ['test']:
            k_sum = 0
            self.subs_dict[train_test_flag] = []

            col_pred_dict = {k : [] for k in np.arange(self.n_targets)}
            for folder_name, k in self.folder_names.iteritems():
                k_sum += k
                sub_df = pd.read_csv(os.path.join(config.OUTPUT_FOLDER, folder_name, 'pred_%s.csv'%train_test_flag))
                sub_df.sort_values('id', inplace = True)

                for i in np.arange(self.n_targets):
                    arr = np.array([sub_df[str(i)].values] * k).transpose()
                    col_pred_dict[i].append(arr)

            # create numpy arrays
            col_pred_dict = {k:np.concatenate(v, axis = 1) for k,v in col_pred_dict.items()}

            final_df = pd.DataFrame({'id' : sub_df['id'].tolist()})
            for i in np.arange(self.n_targets):
                final_df[str(i)] = gmean(col_pred_dict[i], axis = 1)

            self.final_sub[train_test_flag] = final_df

            name = self.save_folder + 'pred_%s.csv'%train_test_flag
            sub = self.final_sub[train_test_flag]
            sub.to_csv(name, index = False)

            if train_test_flag == 'test':
                name = self.save_folder  + '%s.csv'%self.save_folder.split('/')[-2]
                print('\n Saving to %s'%name)
                sub['camera'] = sub.drop('id',1).idxmax(axis = 1).map(self.inverse_mapping_dict)
                sub['fname'] = sub['id']
                print('Submission', sub['camera'].value_counts())
                sub[['fname', 'camera']].to_csv(name, index = False)

Blender(folder_names = folder_names)
