import sys, os
import json
import numpy as np
import pandas as pd
import glob
import generate_sub
from tqdm import tqdm
sys.path.insert(0,'..')
import config

folders = ["nn_24"]
class CsvMerger(object):
    def __init__(self, read_folder):
        self.read_folder  = config.OUTPUT_FOLDER +  read_folder + '/'
        self.predictions_folder = self.read_folder + 'predictions/'
        self.settings = json.load(open(self.read_folder + 'settings.json'))
        self.augs_list = np.arange(len(self.settings['test_augs_list']))
        self.n_labels = 32 #31#12
        # Template('pred_aug_${aug_num}_target_${target_num}')

    def find_nfolds(self):
        self.csv_folds = {}
        self.csv_folds['train'] = glob.glob(self.predictions_folder + 'oof*.csv')
        self.csv_folds['test'] = glob.glob(self.predictions_folder + 'test*.csv')
        assert len(self.csv_folds['train']) == len(self.csv_folds['test'])
        self.n_folds = len(self.csv_folds['train'])

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
        print('Averaging train')
        for j in tqdm(np.arange(self.n_labels)):
            augs_col = [config.pred_col_name_template.substitute(aug_num = aug_num, target_num = j) for aug_num in self.augs_list  ]
            self.train[j] = self.averaging_function(self.train[augs_col].values)
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
        print('Averaging test')
        for j in tqdm(np.arange(self.n_labels)):
            augs_col = ['pred_aug_%d_target_%d_fold_%d'%(aug_num, j, fold) for aug_num in self.augs_list for fold in np.arange(self.n_folds) ]
            self.test[j] = self.averaging_function(self.test[augs_col].values)
            self.test.drop(augs_col, 1, inplace = True)

    def save_to_disk(self):
        print('Saving to %s'% self.read_folder)
        self.test.to_csv( self.read_folder + 'pred_test.csv', index = False)
        self.train.to_csv(self.read_folder + 'pred_train.csv', index = False)

    @staticmethod
    def averaging_function(array):
        power = 4
        res = np.mean(array ** power, axis = 1) ** (1 / float(power))
        return res

    def __call__(self):
        self.find_nfolds()
        self.merge_train_csv()
        self.merge_test_csv()
        self.save_to_disk()

if __name__ == '__main__':
    # folders = ['custom_predictions_%d'%i for i in np.arange(11)]
    bad_folders = []
    for folder in folders:
        print(folder)
        try:
            cm = CsvMerger(folder)
            cm()
            # g = generate_sub.SubGenerator(folder)
            # g()
        except:
            bad_folders.append(folder)
    print("Bad folders : ", bad_folders)
    # print(cm.train.head())
    # print(cm.test.head())