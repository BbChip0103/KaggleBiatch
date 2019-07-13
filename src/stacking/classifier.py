import os
import gc
import json
import logging

import pandas as pd
import numpy as np
import datetime
from sklearn.metrics import accuracy_score

from .module import get_clf
from utils import utils
log = logging.getLogger(__name__)


metric = 'mlogloss'


class Classifier(object):
    def __init__(self, config, data):
        self.config = config
        self.data = data

        #init vars
        self.predictions_list = []
        self.predictions_train = None
        self.evals_result_list = []
        self.all_score_dict = {}
        self.scores_dict = {}
        self.val_acc_list = []

    def do_kfolds(self):
        split_index = int(self.data['train'].shape[0] * 0.8)
        X = self.data['train'][:split_index]
        y = self.data['target'][:split_index]
        X_val = self.data['train'][split_index:]
        y_val = self.data['target'][split_index:]
        payload = {"X": X, "y": y, "X_val": X_val, "y_val": y_val, "test": self.data['test'], "fold_num": 0}

        clf = get_clf(config=self.config)(config=self.config, data=payload)
        pred_results = clf()

        pred_csv = pd.DataFrame(pred_results['test'])
        pred_csv.columns = [str(x) for x in range(self.config['num_classes'])]
        pred_csv['id'] = self.data['test_ids']
        pred_csv.to_csv(os.path.join(self.config['out_folder'], "pred_test.csv"), index=False)

    def calculate_average_score(self):
       # # calculate average score across folds
       #  N = len(self.evals_result_list)
       #
       #  if self.config['mode_stack']['clf_name'] == 'xgb':
       #      list_ = ['train', 'valid']
       #
       #  for name in list_:
       #      self.all_score_dict[name] = []
       #      for fold in range(N):
       #          _ = np.min(self.evals_result_list[fold][name][metric])
       #          self.all_score_dict[name].append(_)
       #      self.scores_dict['std_' + name + '_loss'] = np.std(self.all_score_dict[name])
       #      self.scores_dict['mean_' + name + '_loss'] = np.mean(self.all_score_dict[name])
       #      if name == 'valid':
       #          self.scores_dict['mean_' + name + '_acc'] = np.mean(np.array(self.val_acc_list))
       #          self.scores_dict['std_' + name + '_acc'] = np.std(np.array(self.val_acc_list))
       #
       #  print('\n\n Final scores %s'%str(self.scores_dict))
       #  with open(self.output_folder + 'scores.json', 'w') as fp:
       #      json.dump(self.scores_dict, fp)
       pass

    def predict(self):
        # # blend test predictions across folds
        # predictions_prob = np.median(self.predictions_list, axis = 0)
        #
        # for i, x in enumerate(self.predictions_list):
        #     _ = pd.DataFrame(x)
        #     _.columns = np.arange(31)
        #     _[config.id_col] = self.test[config.id_col].values
        #     _.to_csv(self.output_folder + 'classifier_test_predicitons_fold_%d.csv'%i, index = False)
        #
        # predictions = np.asarray([np.argmax(line) for line in predictions_prob])
        #
        # assert len(predictions) == len(self.test), "Test predictions are shorter than expected"
        # predictions_test = pd.DataFrame({config.id_col: self.test[config.id_col].tolist(), config.target_col: predictions})
        # predictions_prob_df = pd.DataFrame(predictions_prob)
        # predictions_prob_df.columns = [{v:k for k, v in self.mapping_dict.iteritems()}[i] for i in np.arange(31)]
        # predictions_test = pd.concat([predictions_test, predictions_prob_df], axis = 1)
        #
        # for train_test_flag, sub in zip(['train', 'test'], [self.predictions_train, predictions_test]):
        #     name = self.output_folder + 'sub_%s.csv'%train_test_flag
        #     sub[config.target_col] = sub[config.target_col].map({v:k for k, v in self.mapping_dict.iteritems()})
        #     sub.to_csv(name , index = False)
        #     print(train_test_flag, sub[config.target_col].value_counts())
        #
        #     if train_test_flag == 'test':
        #         sub[config.id_col] = sub[config.id_col] + '.wav'
        #         sub[config.target_col] = sub[config.target_col].apply(lambda x: x if x in config.allowed_train_labels else 'unknown')
        #         sub.rename(columns = {config.id_col : 'fname', config.target_col : 'label'}, inplace = True)
        #         sub[['fname', 'label']].to_csv(self.output_folder + 'submission_%s.csv'%self.output_folder.split('/')[-2], index = False)
        pass


    def __call__(self):
        self.do_kfolds()
        self.calculate_average_score()
        self.predict()
