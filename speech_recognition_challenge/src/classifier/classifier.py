# encoding: utf-8
import sys, os   
sys.path.insert(0,'..')
import config
from utils import utils

import pandas as pd
import numpy as np
import datetime
import xgboost as xgb
import module 
import gc
import xgbfir
import json
from sklearn.metrics import accuracy_score

metric = 'mlogloss'

class Classifier(object):
    r"""Class that do ML
    
    ----------
    Args:
    - output_folder = where to save results
    - RS = random state
    - train = train DataFrame
    - test = test DataFrame
    - fold_splits = list of splits for train and validation
    - clf_name = name of classifier
    """
    def __init__(self, output_folder, RS, train, test, fold_splits, mapping_dict, clf_name = 'xgb'):
        self.train = train
        self.test = test
        self.fold_splits = fold_splits
        self.clf_name = clf_name
        self.output_folder = output_folder
        self.mapping_dict = mapping_dict
        self.RS = RS

        #init vars
        self.predictions_list = []
        self.predictions_train = None
        self.evals_result_list = []
        self.all_score_dict = {}
        self.scores_dict = {}
        self.val_acc_list = []

    def do_kfolds(self):
        # for each fold train model
        for i, dict_ in self.fold_splits.iteritems(): 
            print(type(dict_))
            train_ids = dict_['train'] #self.train[config.id_col].tolist() # dict_['train']
            val_ids = dict_['val']  #self.train[config.id_col].tolist() # dict_['val']

            print('\nFold %d / %d'%(i + 1, len(self.fold_splits)))

            # train
            X = self.train[self.train[config.id_col].isin(train_ids)].drop([config.target_col, config.id_col], 1)
            y = self.train[self.train[config.id_col].isin(train_ids)][config.target_col]

            # validation
            X_val = self.train[self.train[config.id_col].isin(val_ids)].drop([config.target_col], 1)
            y_val = self.train[self.train[config.id_col].isin(val_ids)][config.target_col]
            X_val_ids = X_val[config.id_col].tolist()
            X_val.drop(config.id_col, 1, inplace = True)

            # classifying and predicting
            if self.clf_name == 'xgb':
                clf_class = module.XGB
            elif self.clf_name == 'svm':
                clf_class = module.SVM
            elif self.clf_name == 'mlp':
                clf_class = module.MLP
            else:
                raise ValueError("'%s' classifier is not defined!")
            clf = clf_class(output_folder = self.output_folder,  RS = self.RS, 
                                 X = X, y = y, X_val = X_val, y_val = y_val, test = self.test, fold_num = i)
            # appending results
            prediction_train_fold_values_probs, prediction_test_fold_values, evals_result_dict = clf()
            self.predictions_list.append(prediction_test_fold_values)
            self.evals_result_list.append(evals_result_dict)

            prediction_train_fold_values = np.asarray([np.argmax(line) for line in prediction_train_fold_values_probs])
            prediction_train_fold = pd.DataFrame({config.id_col : X_val_ids, config.target_col : prediction_train_fold_values})
            prediction_train_fold_df = pd.DataFrame(prediction_train_fold_values_probs)
            prediction_train_fold_df.columns = [{v:k for k, v in self.mapping_dict.iteritems()}[i] for i in np.arange(31)]
            prediction_train_fold = pd.concat([prediction_train_fold, prediction_train_fold_df], axis = 1)

            # score
            val_acc = accuracy_score(self.train[self.train[config.id_col].isin(X_val_ids)][config.target_col].values, prediction_train_fold_values)
            print('Val accuracy = %f'%val_acc)
            self.val_acc_list.append(val_acc)

            # concatenating OOF train predictions
            if self.predictions_train is None:
                self.predictions_train = prediction_train_fold
            else:
                self.predictions_train = pd.concat([self.predictions_train, prediction_train_fold])

        # assert len(self.train) == len(self.predictions_train) == self.predictions_train[config.id_col].nunique(), \
        #          "OOF error %d, %d , %d"%(len(self.train), len(self.predictions_train), self.predictions_train[config.id_col].nunique())
        print("EBAL", self.evals_result_list)

        # if self.clf_name == 'xgb':
        #     utils.mean_of_xgbfir_folds(self.output_folder, len(self.fold_splits))

    def calculate_average_score(self):
       # calculate average score across folds
        N = len(self.evals_result_list)

        if self.clf_name == 'xgb': 
            list_ = ['train', 'valid']

        for name in list_:
            self.all_score_dict[name] = []
            for fold in range(N):
                _ = np.min(self.evals_result_list[fold][name][metric])
                self.all_score_dict[name].append(_)
            self.scores_dict['std_' + name + '_loss'] = np.std(self.all_score_dict[name]) 
            self.scores_dict['mean_' + name + '_loss'] = np.mean(self.all_score_dict[name]) 
            if name == 'valid':
                self.scores_dict['mean_' + name + '_acc'] = np.mean(np.array(self.val_acc_list))
                self.scores_dict['std_' + name + '_acc'] = np.std(np.array(self.val_acc_list))

        print('\n\n Final scores %s'%str(self.scores_dict))
        with open(self.output_folder + 'scores.json', 'w') as fp:
            json.dump(self.scores_dict, fp)

    def predict(self):
        # blend test predictions across folds
        predictions_prob = np.median(self.predictions_list, axis = 0)

        for i, x in enumerate(self.predictions_list):
            _ = pd.DataFrame(x)
            _.columns = np.arange(31)
            _[config.id_col] = self.test[config.id_col].values
            _.to_csv(self.output_folder + 'classifier_test_predicitons_fold_%d.csv'%i, index = False)

        predictions = np.asarray([np.argmax(line) for line in predictions_prob])

        assert len(predictions) == len(self.test), "Test predictions are shorter than expected"
        predictions_test = pd.DataFrame({config.id_col: self.test[config.id_col].tolist(), config.target_col: predictions})
        predictions_prob_df = pd.DataFrame(predictions_prob)
        predictions_prob_df.columns = [{v:k for k, v in self.mapping_dict.iteritems()}[i] for i in np.arange(31)]
        predictions_test = pd.concat([predictions_test, predictions_prob_df], axis = 1)

        for train_test_flag, sub in zip(['train', 'test'], [self.predictions_train, predictions_test]):
            name = self.output_folder + 'sub_%s.csv'%train_test_flag
            sub[config.target_col] = sub[config.target_col].map({v:k for k, v in self.mapping_dict.iteritems()})
            sub.to_csv(name , index = False)
            print(train_test_flag, sub[config.target_col].value_counts())

            if train_test_flag == 'test':
                sub[config.id_col] = sub[config.id_col] + '.wav'
                sub[config.target_col] = sub[config.target_col].apply(lambda x: x if x in config.allowed_train_labels else 'unknown')
                sub.rename(columns = {config.id_col : 'fname', config.target_col : 'label'}, inplace = True)
                sub[['fname', 'label']].to_csv(self.output_folder + 'submission_%s.csv'%self.output_folder.split('/')[-2], index = False)
                

    def __call__(self):
        self.do_kfolds()
        self.calculate_average_score()
        self.predict()