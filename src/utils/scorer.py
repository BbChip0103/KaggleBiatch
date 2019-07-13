import glob
import os
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm


class NetsScorer(object):
    def __init__(self):
        self.env_file = json.load(open(os.path.join(str(Path.home()), ".kaggle/path.json"), "r"))
        self.nns = glob.glob(os.path.join(self.env_file['output_path'], "protein", "nn", "*", "trainig_log0.csv"))
        self.res = []

    def process_csv(self, csv_path):
        folder_path = os.path.split(csv_path)[0]
        nn_num = os.path.split(folder_path)[-1]
        settings = json.load(open(os.path.join(folder_path, "settings.json"), "r"))
        log = pd.read_csv(csv_path)
        log = log.sort_values("valid_acc", ascending=False)
        log = log.to_dict(orient='records')[0]
        log['name'] = nn_num
        log['net_name'] = settings['net_name']
        log['use_folds'] = settings['use_folds']
        self.res.append(log)

    def __call__(self):
        for nn in tqdm(self.nns):
            self.process_csv(nn)
        self.res = pd.DataFrame(self.res)
        self.res = self.res.sort_values("valid_acc", ascending=False)
        self.res.to_csv(os.path.join(self.env_file['output_path'], "protein", "scores.csv"), index=False)
        print(self.res.head())


if __name__ == "__main__":
    NetsScorer()()
