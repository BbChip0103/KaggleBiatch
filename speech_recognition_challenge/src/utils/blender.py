import pandas as pd
import uuid
import numpy as np
import fegolib as fgl
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
import json
import glob
import joblib
from tqdm import tqdm
import sys, os
sys.path.insert(0,'..')
import config
import utils

#experiments
folder_names =     { 
                    "nn_16": 1,
                    "nn_19" : 1,
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
        self.save_folder = fgl.utils.path_that_not_exist(config.OUTPUT_FOLDER + 'blnd/', create = True)

        self.folder_names = folder_names

        self.mode =  mode
        print('Selected mode : %s'%self.mode)
        #drop duplicates from dict
        result = {}
        for key, value in self.folder_names.items():
            if key not in result.keys():
                result[key] = value
        self.folder_names = result
        print(self.folder_names)
        
        # with open(self.save_folder + 'blend_dict.json', 'w') as fp:
        #     json.dump({'mode' : self.mode, 'subs_dict' : self.folder_names}, fp)
        
        # self.train = pd.read_json(config.DATA_FOLDER + 'train.json')
        # self.train = self.train[['id', config.target_col]]
        self.process_data()

        # self.calculate_score()

    def process_data(self):
        self.subs_dict = {}
        self.final_sub = {}
        RND = self.save_folder.split('/')[-1]

        for train_test_flag in ['train', 'test']:
            final_df = None
            k_sum = 0
            self.subs_dict[train_test_flag] = []
            for folder_name, k in self.folder_names.iteritems():
                k_sum += k
                sub_df = pd.read_csv(config.OUTPUT_FOLDER + folder_name + '/pred_%s.csv'%train_test_flag)

                if 'neural_clf' not in folder_name:
                    # le = joblib.load(config.OUTPUT_FOLDER + folder_name + '/' + 'label_encoder.dump')
                    for i in np.arange(32):
                        new_col_name = {v:k for k, v in config.mapping_dict.iteritems()}[i] 
                        sub_df[str(i)] = (sub_df[str(i)] ** 0.5)* k
                        sub_df.rename(columns = {str(i) : new_col_name + '_' + folder_name}, inplace = True)
                else:
                    for i in np.arange(32):
                        new_col_name = {v:k for k, v in config.mapping_dict.iteritems()}[i] 
                        sub_df[str(i)] = (sub_df[str(i)] ** 0.5)* k
                        sub_df.rename(columns = {str(i) : new_col_name+ '_' + folder_name}, inplace = True)

                if isinstance(final_df, pd.DataFrame):
                    final_df = final_df.merge(sub_df, on = 'id')
                else:
                    final_df = sub_df

            for i in config.mapping_dict.keys():
                print(i)
                cols = [x for x in final_df.columns.tolist() if str(i) in x]
                final_df[str(i)] = (final_df[cols].values ** 4).mean(axis = 1) ** (0.25)
                final_df.drop(cols, 1, inplace = True)

            self.final_sub[train_test_flag] = final_df
        
            name = self.save_folder + 'pred_%s.csv'%train_test_flag
            sub = self.final_sub[train_test_flag]
            sub['target'] = sub.drop('id',1).idxmax(axis = 1)
            sub.to_csv(name, index = False)

            if train_test_flag == 'test':
                name = self.save_folder  + 'blend_%s.csv'%self.save_folder.split('/')[-2]
                print('\n Saving to %s'%name)
                sub['label'] = sub['target']
                sub['fname'] = sub['id'] + '.wav'
                sub['label'] = sub['label'].apply(lambda x: x if x in config.allowed_train_labels else 'unknown')
                assert sub['label'].nunique() == len(config.allowed_train_labels) + 1
                print('Submission', sub['label'].value_counts())
                sub[['fname', 'label']].to_csv(name, index = False)

    # def calculate_score(self):
    #     #calculate best single model score

    #     self.score_single = 10.0
    #     for df in self.subs_dict['train']:
    #         c_col = [x for x in df.columns.tolist() if config.target_col + '_' in x]
    #         assert len(c_col) == 1
    #         c_col = c_col[0]

    #         merged_single = self.train.copy().merge(df, on = 'id')
    #         _ = metric(merged_single[config.target_col].values, merged_single[c_col].values)
    #         if _ < self.score_single:
    #             self.score_single = _

    #     merged = self.train.merge(self.final_sub['train'], on = 'id', suffixes=('_gt', '_pr'))
        
    #     assert len(self.train) == len(merged), 'Len of merged data is invalid'
    #     self.score = metric(merged[config.target_col + '_gt'].values, merged[config.target_col + '_pr'].values)

    #     if len(self.subs_dict['train']) != len(self.subs_dict['test']):
    #         print('Score is not accurate, because of not all train data')
    #     print("Blender score = %f, Best Single Model score = %f"%(self.score, self.score_single))

    #     with open(self.save_folder + 'results.json', 'w') as fp:
    #         json.dump({'blender_score' : self.score, 'bsm' : self.score_single}, fp)

Blender(folder_names = folder_names, mode = 'mean')
