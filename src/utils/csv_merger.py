import sys, os
import json
import numpy as np
import pandas as pd
import glob
import generate_sub
sys.path.insert(0,'..')
import config
from scipy.stats.mstats import gmean


folder = 'nn/37'
# merges aug
# and then average voting for each split to from a picture

class CsvMerger(object):
    def __init__(self, read_folder):
        self.read_folder  = os.path.join(config.OUTPUT_FOLDER, read_folder)
        self.predictions_folder = os.path.join(self.read_folder, 'predictions')
        self.settings = json.load(open(os.path.join(self.read_folder, 'settings.json')))
        self.augs_list = np.arange(len(self.settings['test_augs_list']))
        self.n_labels = 340
        # Template('pred_aug_${aug_num}_target_${target_num}')

    def find_nfolds(self):
        self.csv_folds = {}
        self.csv_folds['train'] = glob.glob(os.path.join(self.predictions_folder, 'oof_prediction*.csv'))
        self.csv_folds['test'] = glob.glob(os.path.join(self.predictions_folder, 'test_prediction*.csv'))
        # assert len(self.csv_folds['train']) == len(self.csv_folds['test'])
        self.n_folds = len(self.csv_folds['test'])

        print('Found %d folds'%self.n_folds)

    def merge_train_csv(self):
        print('Merging train')
        self.train = None
        for csv_name in self.csv_folds['train']:
            csv = pd.read_csv(csv_name)
            if self.train is None:
                self.train = csv
            else:
                self.train = pd.concat([self.train, csv])

        for j in np.arange(self.n_labels):

            augs_col = [config.pred_col_name_template.substitute(aug_num = aug_num, target_num = j) for aug_num in self.augs_list  ]
            self.train[j] = gmean(self.train[augs_col].values, axis=1)
            self.train.drop(augs_col, 1, inplace = True)

    def merge_test_csv(self):
        print('Merging test')
        self.test = None
        for csv_name in self.csv_folds['test']:
            csv = pd.read_csv(csv_name)
            fold_num = csv_name.split('fold_')[-1].split('.')[0]
            csv.rename(columns = {k : k + '_fold_%s'%fold_num  for k in csv.columns.tolist() if k != 'id' }, inplace = True)

            if self.test is None:
                self.test = csv
            else:
                self.test = self.test.merge(csv, on = 'id')
                assert len(self.test) == len(csv)
        # print(self.test.columns.tolist())
        for j in np.arange(self.n_labels):
            augs_col = ['pred_aug_%d_target_%d_fold_%d'%(aug_num, j, fold) for aug_num in self.augs_list for fold in np.arange(self.n_folds) ]
            self.test[j] = gmean(self.test[augs_col].values, axis = 1)
            self.test.drop(augs_col, 1, inplace = True)

    @staticmethod
    def tiles_voting(df):
        # df['new_id'] = df['id'].apply(lambda x: x.split('_split')[0])
        new_df = df.groupby('id').agg({c : 'median' for c in np.arange(340)}).reset_index()
        # new_df.rename(columns = {'new_id' : 'id'}, inplace = True)
        return new_df

    def save_to_disk(self):
        print('Saving to %s'% self.read_folder)
        # assert len(self.train) == config.LEN_DFS['train']
        # assert len(self.test) == config.LEN_DFS['test']
        self.test.to_csv(os.path.join(self.read_folder, 'pred_test.csv'), index = False)
        # self.train.to_csv(os.path.join(self.read_folder, 'pred_train.csv'), index = False)

    def __call__(self):
        self.find_nfolds()
        # self.merge_train_csv()
        self.merge_test_csv()
        self.test = self.tiles_voting(self.test)
        # self.train = self.tiles_voting(self.train)
        self.save_to_disk()

if __name__ == '__main__':
    # cm = CsvMerger(folder)
    # cm()
    # # print(cm.train.head())
    # print(cm.test.head())

    import pandas
    from pathlib import Path
    import os
    import pandas as pd
    import json
    from pathlib import Path
    category_mapping = json.load(open(os.path.join(str(Path.home()), ".kaggle_splits", "doodle", "category_mapping.json"), "r"))

    path = os.path.join(Path.home(), "output", "doodle", "nn", "37", "pred_test.csv")
    preds = pd.read_csv(path)

    df = preds.set_index("id")
    k = 3
    res = pd.DataFrame({n: df.T[column].nlargest(k).index.tolist() for n, column in enumerate(df.T)}).T
    inversed_dict = {v:"_".join(k.split(" ")) for k,v in category_mapping.items()}

    for i in range(k):
        if i == 0:
            res['word'] =  res[i].apply(lambda x: inversed_dict[int(x)])
        else:
            res['word'] = res['word'] + " " + res[i].apply(lambda x: inversed_dict[int(x)])
    res['key_id'] = preds['id']
    res[['key_id', "word"]].to_csv(os.path.join(Path.home(), "output", "doodle", "nn", "37", "sub.csv"),  index= False)
