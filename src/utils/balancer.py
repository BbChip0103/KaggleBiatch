import sys
import os
import json
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0,'..')
import config


read_folder = "blnd/1"


class Balancer(object):
    def __init__(self, read_folder, tta=True):
        # Clases = list of labels
        self.read_folder = read_folder
        self.tta = tta
        self.num_classes = 340

    def _get_predicts(self, predicts, coefficients):
        return torch.einsum("ij,j->ij", (predicts, coefficients))

    def _get_labels_distribution(self, predicts, coefficients):
        predicts = self._get_predicts(predicts, coefficients).float()
        labels = predicts.argmax(axis=-1)
        counter = torch.bincount(labels, minlength=predicts.shape[1])
        return counter

    def _compute_score_with_coefficients(self, predicts, coefficients):
        counter = self._get_labels_distribution(predicts, coefficients).float()
        counter = counter * 100 / len(predicts)
        max_scores = torch.ones(len(coefficients)).cuda().float() * 100 / len(coefficients)
        result, _ = torch.min(torch.cat([counter.unsqueeze(0), max_scores.unsqueeze(0)], dim=0), dim=0)
        return float(result.sum().cpu())

    def get_coeffs(self, predicts, coefficients, alpha=0.001, iterations=100):
        best_coefficients = coefficients.clone()
        best_score = self._compute_score_with_coefficients(predicts, coefficients)

        for _ in tqdm.trange(iterations):
            counter = self._get_labels_distribution(predicts, coefficients)
            label = int(torch.argmax(counter).cpu())
            coefficients[label] -= alpha
            score = self._compute_score_with_coefficients(predicts, coefficients)
            if score > best_score:
                best_score = score
                best_coefficients = coefficients.clone()

        return best_coefficients.numpy()

    def __call__(self):
        read_folder = os.path.join(config.OUTPUT_FOLDER, self.read_folder)

        csv_name = "pred_test.csv"
        if not self.tta:
            csv_name = os.path.join("predictions", "test_aug_0_fold_0.csv")
        path = os.path.join(read_folder, csv_name)
        print("Using name: ", path)
        csv = pd.read_csv(path)
        # ###
        # print(csv.columns.tolist())
        # assert False
        # csv = csv.drop(columns=["Unnamed: 0", "key_id", "word"])
        csv = csv.drop(columns=["id"])
        ###

        pred_cols_names = [k for k in csv.columns.tolist() if k != 'id']
        predicts = csv[pred_cols_names].values

        balance_dict = {}
        tta = 0
        balance_dict[tta] = self.get_coeffs(predicts=predicts, iterations=100000)
        for i, c in enumerate(pred_cols_names[tta * self.num_classes: (tta+1) * self.num_classes]):
            csv[c] = csv[c] * balance_dict[tta][i]

        with open(os.path.join(read_folder, "balance_coefs.json"), "w") as f:
            json.dump(balance_dict, f)

        save_name = "pred_test_balanced.csv"
        if not self.tta:
            save_name = "pred_test_balanced_no_tta.csv"
        csv.to_csv(os.path.join(read_folder, save_name), index=False)


if __name__ == "__main__":
    N = 1000
    # preds = pd.read_csv(p)
    read_folder  = os.path.join(config.OUTPUT_FOLDER,  read_folder)
    B = Balancer(read_folder, tta=True)
    B()
#     coefficients = B(preds)
#     # labels, _ = _get_labels_distribution(preds, coefficients)
#     # labels = map(lambda label: CLASSES[label], labels)
#     print(coefficients)
