import os
import logging

import torch
import xgboost as xgb
import lightgbm as lgb
import xgbfir
log = logging.getLogger(__name__)


N_threads = 8
metric = 'multi:softprob'


class _module(object):
    def __init__(self, config, data):
        self.config = config
        self.data = data
        self.results = {}

    def __call__(self):
        self._fit()
        weight_name = f'{self.config["mode_stack"]["clf_name"]}_weight_fold{self.data["fold_num"]}.model'
        torch.save({"clf": self.clf, "params": self.params}, os.path.join(self.config["out_folder"], weight_name))
        self._predict()
        self._additional_task()
        results = {'val': self.prediction_train, 'test': self.prediction_test}
        return results

    def _additional_task(self):
        pass


class XGB(_module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.config['competition_type'] == "multiclass":
            obj = 'multi:softprob'
        else:
            raise NotImplementedError

        self.params = {
            'objective': obj,
            'eta': self.config['mode_stack']['lr'],
            'silent' : True,
            'max_depth' : 4,
            'subsample' : 0.8,
            'colsample_bytree' : 0.8,
            'nthread': self.config['mode_stack']['N_threads'],
            # 'eval_metric': metric,
            'alpha': 0,
            # 'max_delta_step':0.7
            'num_class' : self.config['num_classes'],
            'lambda': 1,
            'min_split_loss' : 1,
            'seed' : self.config['mode_stack']["RS"],
            'eval_metric' : "mlogloss",
            "three_method" : "hist",
        }

    def _fit(self):
        train_data = xgb.DMatrix(self.data["X"], label = self.data["y"])
        val_data = xgb.DMatrix(self.data["X_val"], label = self.data["y_val"])

        watchlist = [(train_data, 'train'), (val_data, 'valid')]
        self.evals_result_dict = {}
        self.clf = xgb.train(self.params,
                             train_data,
                             self.config['mode_stack']['num_rounds'],
                             watchlist,
                             early_stopping_rounds=self.config['mode_stack']['early_stopping_rounds'],
                             maximize=False,
                             verbose_eval=10,
                             evals_result=self.evals_result_dict )

    def _predict(self):
        self.prediction_test = self.clf.predict(xgb.DMatrix(self.data['test']))
        self.prediction_train = self.clf.predict(xgb.DMatrix(self.data["X_val"]))

    def _additional_task(self):
        save_path = os.path.join(self.config["out_folder"], f'xgbfir_{self.data["fold_num"]}.xlsx')
        xgbfir.saveXgbFI(self.clf, OutputXlsxFile=save_path)


class LGB(_module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.config['competition_type'] == "multiclass":
            obj = 'multiclass'
        else:
            raise NotImplementedError

        self.params = {
            'boosting_type': 'gbdt',
            'objective': obj,
            'nthread': self.config['mode_stack']['N_threads'],
            'num_class' : self.config['num_classes'],
            "learning_rate": self.config['mode_stack']['lr'],
            # "max_depth": 9,
            'seed' : self.config['mode_stack']["RS"],
            "three_method" : "hist",
            # "max_bin": 255,
            "metric": "multi_logloss",
            "verbose_eval": -1,
            # "num_leaves":31
        }

    def _fit(self):
        train_data = lgb.Dataset(self.data["X"], self.data["y"])
        val_data = lgb.Dataset(self.data["X_val"], self.data["y_val"])

        watchlist = [(train_data, 'train'), (val_data, 'valid')]
        self.clf = lgb.train(self.params,
                             train_data,
                             num_boost_round=self.config['mode_stack']['num_rounds'],
                             valid_sets=val_data,
                             early_stopping_rounds=self.config['mode_stack']['early_stopping_rounds'])

    def _predict(self):
        self.prediction_test = self.clf.predict(self.data['test'])
        self.prediction_train = self.clf.predict(self.data["X_val"])


def get_clf(config):
    if config['mode_stack']['clf_name'] == 'xgb':
        log.info("Initializing XGB")
        clf = XGB
    elif config['mode_stack']['clf_name'] == "lgb":
        log.info("Initializing LGB")
        clf = LGB
    else:
        raise ValueError("'%s' classifier is not defined!")
    return clf
