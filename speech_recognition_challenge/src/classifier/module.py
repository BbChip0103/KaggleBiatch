# encoding: utf-8
import sys, os   
sys.path.insert(0,'..')
import config
import joblib

import xgboost as xgb
import xgbfir
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

N_threads = 8
metric =  'multi:softprob'
num_iters = 4000
    
class _module(object):
    def __init__(self, output_folder, RS, X, y, X_val, y_val, test, fold_num):
        self.output_folder = output_folder
        self.X = X
        self.y = y
        self.X_val = X_val
        self.y_val = y_val
        self.test = test
        self.fold_num = fold_num
        self.RS = RS
class XGB(_module):
    def __init__(self, output_folder, RS, X, y, X_val, y_val, test, fold_num):
        super(XGB, self).__init__(output_folder, RS, X, y, X_val, y_val, test, fold_num)

        self.params = {
            'objective': 'multi:softprob',
            'eta': 0.01,
            'silent' : True,
            'max_depth' : 4,
            'subsample' : 0.8,
            'colsample_bytree' : 0.8,
            'nthread': N_threads,
            # 'eval_metric': metric,
            'alpha': 0,
            # 'max_delta_step':0.7
            'num_class' : 31,
            'lambda': 1,
            'min_split_loss' : 1,
            'seed' : self.RS,
            'eval_metric' : "mlogloss",
            "three_method" : "hist",
        }

    def _fit(self):
        # train classifier on single fold
        train_data = xgb.DMatrix(self.X, label = self.y)
        val_data = xgb.DMatrix(self.X_val, label = self.y_val)

        watchlist = [(train_data, 'train'), (val_data, 'valid')]
        self.evals_result_dict = {}
        self.clf = xgb.train(self.params, train_data, num_iters, watchlist, early_stopping_rounds = 100, 
                    # feval = utils.gini_xgb, 
                    maximize = False, verbose_eval = 10, evals_result = self.evals_result_dict )

        joblib.dump(self.clf, self.output_folder + 'xgb_weight_fold%d.model'%self.fold_num)

    def _predict(self):
        # predict OOF and test
        self.prediction_test = self.clf.predict(xgb.DMatrix(self.test.drop(config.id_col, 1)))
        self.prediction_train = self.clf.predict(xgb.DMatrix(self.X_val))

    def _save_feature_importance(self):
        xgbfir.saveXgbFI(self.clf, OutputXlsxFile = self.output_folder + 'xgbfir_%d.xlsx'%self.fold_num)

    def __call__(self):
        self._fit()
        self._predict()
        self._save_feature_importance()
        return self.prediction_train, self.prediction_test, self.evals_result_dict

class SVM(_module):
    def __init__(self, output_folder, RS, X, y, X_val, y_val, test, fold_num):
        super(SVM, self).__init__(output_folder, RS, X, y, X_val, y_val, test, fold_num)

        self.params = {'random_state' : self.RS,
                       'kernel' : 'rbf',
                       'probability' : True,
                       'tol' : 1e-4,
                       'verbose' : False,
                      }
    
    def _fit(self):
        # train classifier on single fold
        self.clf = SVC(**self.params)
        self.clf.fit(self.X, self.y)
        self.evals_result_dict = {}

    def _predict(self):
        # predict OOF and test
        self.prediction_test = self.clf.predict(self.test.drop(config.id_col, 1))
        self.prediction_train = self.clf.predict(self.X_val)

    def __call__(self):
        self._fit()
        self._predict()
        return self.prediction_train, self.prediction_test, self.evals_result_dict

class MLP(_module):
    def __init__(self, output_folder, RS, X, y, X_val, y_val, test, fold_num):
        super(MLP, self).__init__(output_folder, RS, X, y, X_val, y_val, test, fold_num)

        self.params = {'activation' : 'relu', 
                       'alpha' : 1e-05, 
                       'batch_size' : 1024,
                       'beta_1' : 0.9, 
                       'beta_2' : 0.999, 
                       'early_stopping'  : True,
                       'epsilon' : 1e-08, 
                       'hidden_layer_sizes' : (10, 2), 
                       'learning_rate' : 'adaptive',
                       'learning_rate_init' : 0.001, 
                       'max_iter' : 1000, 
                       'momentum' : 0.9,
                       'nesterovs_momentum' : True, 
                       'random_state' : self.RS, 
                       'shuffle' : True,
                       'solver' : 'sgd', 
                       'tol' : 0.0001, 
                       'validation_fraction':  0.1, 
                       'verbose' : True,
                       'warm_start' : False
                      }
    
    def _fit(self):
        # train classifier on single fold
        self.clf = MLPClassifier(**self.params)
        self.clf.fit(self.X, self.y)
        self.evals_result_dict = {}

    def _predict(self):
        # predict OOF and test
        self.prediction_test = self.clf.predict(self.test.drop(config.id_col, 1))
        self.prediction_train = self.clf.predict(self.X_val)

    def __call__(self):
        self._fit()
        self._predict()
        return self.prediction_train, self.prediction_test, self.evals_result_dict
