import os
from pathlib import Path
import json
import glob

import numpy as np
from tqdm import tqdm
import pandas as pd
import torch

folders = ["nn/70", "nn/94"]


def convert_single(folder):
    env_file = json.load(open(os.path.join(str(Path.home()), ".kaggle/path.json"), "r"))
    out_path = os.path.join(env_file['output_path'], "doodle", folder)
    preds_path = glob.glob(os.path.join(out_path, "pred_test.csv")) + glob.glob(os.path.join(out_path, "pred_train.csv"))
    for p in preds_path:  
        print(f"Reading {p}")
        df = pd.read_csv(p)
        df.set_index("id", inplace=True)
        values = df.values
        ids = df.index.tolist()
        res = {"id": ids, "pred": values.astype(np.float32)}

        torch_path = list(os.path.split(p))
        torch_path[-1] = torch_path[-1].split(".csv")[0] + ".pth"
        torch.save(res, os.path.join(*torch_path))


if __name__ == "__main__":
    for f in tqdm(folders):
        convert_single(f)
