# encoding: utf-8
import sys, os   
sys.path.insert(0,'..')
import config
import utils.utils as utils
from sklearn.utils import shuffle
import gc

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

nn_folders = ['nn_4', 'nn_5']

class CLFDataset(object):
    def __init__(self, nn_folders):
        self.nn_folders = nn_folders

    def read_data(self):
        self.df_dict = {}

        for train_test_flag in ['train', 'test']:
            final_df = None
            self.ohe_cols = []
            print('Reading %s'%train_test_flag)
            for folder_name in tqdm(self.nn_folders):
                sub_df = pd.read_csv(config.OUTPUT_FOLDER + folder_name + '/pred_%s.csv'%train_test_flag)

                print("INT", int(folder_name.split("_")[-1]))
                if (int(folder_name.split("_")[-1]) <= 11):
                    le = joblib.load(config.OUTPUT_FOLDER + folder_name + '/' + 'label_encoder.dump')
                    sub_df.rename(columns = {str(i) : le.inverse_transform(i) for i in np.arange(31)}, inplace = True)
                else:
                    inverse_mapping_dict = {v:k for k, v in config.mapping_dict.iteritems()}
                    sub_df.rename(columns = {str(i) : inverse_mapping_dict[i] for i in np.arange(31)}, inplace = True)
                
                # top_col = 'top1_cat_%s'%folder_name
                # sub_df[top_col] = sub_df.drop(config.id_col, 1).idxmax(axis=1)
                # self.ohe_cols.append(top_col)
                #sub_df.rename(columns = {c : c + '_' + folder_name for c in sub_df.drop([top_col, config.id_col], 1).columns.tolist()}, inplace = True)              

                sub_df.rename(columns = {c : c + '_' + folder_name for c in sub_df.drop([config.id_col], 1).columns.tolist()}, inplace = True)              

                if isinstance(final_df, pd.DataFrame):
                    final_df = final_df.merge(sub_df, on = 'id')
                else:
                    final_df = sub_df

            assert len(config.categories.keys()) == 31
            self.df_dict[train_test_flag] = final_df


    def get_splits(self, folder):
        self.splits = {}
        for i in np.arange(5):
            self.splits[i] = {}
            for c in ['train', 'val']:
                self.splits[i][c] = utils.read_list_from_file(config.OUTPUT_FOLDER + folder + '/splits/%s_%d.txt'%(c, i))

    # def fe(self):
    #     self.test[config.target_col] = np.nan
    #     self.df = pd.concat([self.test, self.train])
    #     self.one_hot_encode(cat_feats_list = self.ohe_cols)
    #     self.train = self.df[self.df[config.target_col].notnull()].copy()
    #     self.train = shuffle(self.train)
    #     self.train.reset_index(drop = True, inplace = True)
    #     self.test = self.df[self.df[config.target_col].isnull()].copy()
    #     self.test.drop(config.target_col, 1, inplace = True)
    #     del self.df; gc.collect()

    # def one_hot_encode(self, cat_feats_list ,drop = True):
    #     self.one_hot_feats = []
    #     print('One hot encoding categorical.')
    #     for c in (cat_feats_list):
    #         try:
    #             one_hot = pd.get_dummies(self.df[c], dummy_na = True, drop_first = True)
    #             modified_col_names = ['one_hot_' + c + '_' + str(col) for col in one_hot.columns.tolist()]
    #             one_hot.columns = modified_col_names
    #             self.one_hot_feats += modified_col_names
    #             if drop:
    #                 self.df.drop(c, 1, inplace = True)
    #             self.df = pd.concat([self.df, one_hot], axis = 1)
    #             del one_hot; gc.collect()
    #         except: pass


    def __call__(self):
        self.read_data()
        self.df_dict['train'][config.target_col] =  self.df_dict['train'][config.id_col].apply(lambda x: x.split("_")[-1])
        self.df_dict['train'][config.target_col] = self.df_dict['train'][config.target_col].apply(lambda x: x if x in config.allowed_train_labels else "unknown")
        self.df_dict['train'][config.target_col] =  self.df_dict['train'][config.target_col].map(config.mapping_dict_12_with_unknown )
        print(self.df_dict['train'].head())
        self.train = self.df_dict['train'].copy()
        self.test = self.df_dict['test'].copy()
        del self.df_dict

        # self.fe()

        print(self.train.columns.tolist())

        self.get_splits(self.nn_folders[-1])

        return self.train, self.test, self.splits

if __name__ == '__main__':
    clf_ds = CLFDataset(nn_folders = nn_folders)
    train, test, splits = clf_ds()

    print(train.head())    
    print(test.head())
    