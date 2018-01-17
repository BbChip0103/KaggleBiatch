import os
import pandas as pd
import glob
from tqdm import tqdm
from scipy.io import wavfile
from scipy import signal
import numpy as np
import utils
import sys, os

sys.path.insert(0,'..')
import config


all_train_files = glob.glob(config.DATA_FOLDER + 'train/audio/*/*wav')
all_test_files = glob.glob(config.DATA_FOLDER + 'test/audio/*wav')

test_img_max_list = []
test_img_min_list = []

for file_path in tqdm(all_test_files):
    _, img = utils.read_wav(file_path)
    
    test_img_min_list.append(img.min())
    test_img_max_list.append(img.max())


print(np.min(test_img_min_list), np.max(test_img_max_list))
#train
(-23.02585, 16.476927)