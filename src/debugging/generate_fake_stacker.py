from pathlib import Path
import os

import torch
import numpy as np


for i in range(2):
    for train_test_flag in ['train', "test"]:
        shuffle(ids)
        pred = {"id": ids, "pred": np.random.rand(N, 10).astype(np.float32)}
        path = os.path.join(str(Path.home()), "output", "dummy_competition", "nn", f"{i}", f"pred_{train_test_flag}.pth")
        torch.save(pred, path)
