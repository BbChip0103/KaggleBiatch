import os
import json
import argparse
from pathlib import Path

import torch
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
from tqdm import tqdm

from utils import utils
from dataset.dataset import Dataset


class Submission(object):
    def __init__(self):
        self.get_args()
        self.get_config()
        self.set_logger()

    def get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--folder", help="Flder, e.g. nn/15")
        parser.add_argument("--threshold", help="mean or by_class")

        args = parser.parse_args()
        self.config = {"process_folder": args.folder, "threshold_mode": args.threshold}

    def get_config(self):
        env_file = json.load(open(os.path.join(str(Path.home()), ".kaggle/path.json"), "r"))
        competition_file = json.load(open(os.path.join(os.getcwd(), "competition.json"), "r"))
        out_folder = os.path.join(env_file['output_path'],
                                  competition_file['competition_name'],
                                  self.config['process_folder'])
        self.config.update(json.load(open(os.path.join(out_folder, "settings.json"), "r")))
        self.config['out_folder'] = out_folder
        self.config['predictions_folder'] = os.path.join(out_folder, "predicitons")
        self.config['data_folder'] = os.path.join(env_file['data_path'],
                                                  self.config['competition_data_folder'])
        self.config['splits_path'] = os.path.join(str(Path.home()),
                                                  ".kaggle_splits",
                                                  self.config['competition_name'])
        self.config.update(competition_file)

    def set_logger(self):
        self.log = utils.set_logger(out_folder=self.config['out_folder'], name="sub_log")
        self.log.info(f"{self.config}")

    def get_data(self):
        self.ds = Dataset(config=self.config)
        self.ds()

    def calculate_f1(self, thres, col=None):
        y_pred = self.data['val']['pred'].numpy() > thres
        y_true = np.empty_like(y_pred)
        for i, k in enumerate(self.data['val']['id']):
            y_true[i, :] = self.ds.train[k]['target']
        f1_results = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
        return f1_results

    def calculate_f1_by_class(self, thres, col):
        y_pred = self.data['val']['pred'][:, col].numpy() > thres
        y_true = np.empty_like(y_pred)
        for i, k in enumerate(self.data['val']['id']):
            y_true[i] = self.ds.train[k]['target'][col]
        f1_results = f1_score(y_true=y_true, y_pred=y_pred, average='binary')
        return f1_results

    def get_optimal_thrs(self, col=None):
        score = 0
        opt_thres = 0
        for thres in tqdm(range(100)):
            thres = thres / 100.0
            score_tmp = self.score_f1(thres, col)
            if score_tmp > score:
                score = score_tmp
                opt_thres = thres
        return opt_thres, score

    def get_optimal_thrs_by_class(self):
        self.log.info("Optimizing threshold by class")
        opt_threshold = np.empty(self.config['num_classes'])
        scores = np.empty(self.config['num_classes'])
        for i in range(self.config['num_classes']):
            opt_threshold[i], scores[i] = self.get_optimal_thrs(col=i)
        scores_f1 = self.calculate_f1(opt_threshold)
        return opt_threshold, scores_f1

    def process_thres(self):
        self.log.info("Optimizing threshold")
        if self.config['threshold_mode'] == "mean":
            self.score_f1 =  self.calculate_f1
            self.opt_thres, score = self.get_optimal_thrs()
        elif self.config['threshold_mode'] == "by_class":
            self.score_f1 =  self.calculate_f1_by_class
            self.opt_thres, score = self.get_optimal_thrs_by_class()
            # self.opt_thres = np.min([self.opt_thres, 0.27 * np.ones(self.config['num_classes'])], axis=0)
        else:
            raise NotImplementedError
        self.log.info(f"Optimal thres is {self.opt_thres} and val f1_score = {score}")

    def get_preds(self):
        self.data = {}
        for t in ['test', 'val']:
            path = os.path.join(self.config['predictions_folder'], f"{t}_prediction_fold_0.pth")
            self.data[t] = torch.load(path)

    def generate_sub(self):
        pred_list = []
        for line in self.data['test']['pred'].numpy():
            s = ' '.join(list([str(i) for i in np.nonzero(line > self.opt_thres)[0]]))
            if s == "":
                s = str(line.argmax())
            pred_list.append(s)
        sub = pd.DataFrame({self.config['competition_id_col']: self.data['test']['id'],
                            self.config['competition_predict_target_col']: pred_list})
        save_path = os.path.join(self.config['out_folder'], f"sub_{self.config['threshold_mode']}.csv")
        sub.to_csv(save_path, index=False)

    def __call__(self):
        self.get_data()
        self.get_preds()
        self.process_thres()
        self.generate_sub()

if __name__ == '__main__':
    Submission()()
