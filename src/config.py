import os
from multiprocessing import cpu_count
import torch
import warnings
import numpy as np
import random
from string import Template
import glob
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
cudnn.deterministic = True
warnings.filterwarnings("ignore")


#select which gpu to use
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#-------------------------PATHS--------------------------------
DRIVE = '/mnt/sdc1/'
SSD = '/home/tony/ssd_buffer/'
DATA_FOLDER = SSD + 'kaggle_speech/'
OUTPUT_FOLDER = DRIVE + 'cloud/kaggle_output/kaggle_speech_output/'
OUTPUT_FOLDER_PSEUDO_DATA = OUTPUT_FOLDER + '_a_pseudo/'


#GGC
# DRIVE = '/home/antonagoo/'
# SSD = '/home/antonagoo/'
# DATA_FOLDER = SSD + 'speech_data/'
# OUTPUT_FOLDER = DRIVE + 'speech_output/'
# OUTPUT_FOLDER_PSEUDO_DATA = OUTPUT_FOLDER + '_a_pseudo/'

#--------------------------------------------------------------

#------------------------CONSTANTS-----------------------------
THREADS = cpu_count() 
USE_CUDA = torch.cuda.is_available()
#-----------------------------------------------------------------


id_col = 'id'
target_col = 'target'
sub_target_col = 'label'
tmp_col = 'target_tmp'
xgbfir_folder = 'xgbfir'
gbt_folder = 'gbt'
names_dict = {'train': 'train_pred_oof', 'test' : 'test_pred'}


pred_col_name_template =  Template('pred_aug_${aug_num}_target_${target_num}')
#--------------------------IMGS--------------------

resolution = {'w' : 161, 'h' : 99}


#-------------------------------------------------------
categories = {  'bed': 1713,
                'bird': 1731,
                'cat': 1733,
                'dog': 1746,
                'down': 2359,
                'eight': 2352,
                'five': 2357,
                'four': 2372,
                'go': 2372,
                'happy': 1742,
                'house': 1750,
                'left': 2353,
                'marvin': 1746,
                'nine': 2364,
                'no': 2375,
                'off': 2357,
                'on': 2367,
                'one': 2370,
                'right': 2367,
                'seven': 2377,
                'sheila': 1734,
                'silence' : 2372,
                'six': 2369,
                'stop': 2380,
                'three': 2356,
                'tree': 1733,
                'two': 2373,
                'up': 2375,
                'wow': 1745,
                'yes': 2377,
                'zero': 2376}
#-------------------------------------------------------
allowed_train_labels = ['down',
                        'go',
                        'left',
                        'no',
                        'off',
                        'on',
                        'right',
                        'silence',
                        'stop',
                        'up',
                        'yes']


mapping_dict = {'bed': 11,
                'bird': 12,
                'cat': 13,
                'dog': 14,
                'down': 0,
                'eight': 15,
                'five': 16,
                'four': 17,
                'go': 1,
                'happy': 18,
                'house': 19,
                'left': 2,
                'marvin': 20,
                'nine': 21,
                'no': 3,
                'off': 4,
                'on': 5,
                'one': 22,
                'right': 6,
                'seven': 23,
                'sheila': 24,
                'silence': 7,
                'six': 25,
                'stop': 8,
                'three': 26,
                'tree': 27,
                'two': 28,
                'up': 9,
                'wow': 29,
                'yes': 10,
                'zero': 30, 
                'unknown' : 31}

mapping_dict_12_with_unknown = {'down': 0,
                                'go': 1,
                                'left': 2,
                                'no': 3,
                                'off': 4,
                                'on': 5,
                                'right': 6,
                                'silence': 7,
                                'stop': 8,
                                'up': 9,
                                'yes': 10,
                                'unknown' : 11}

noise_files = glob.glob(DATA_FOLDER + 'train/audio/_background_noise_/*.wav')


LEN_DFS = {'test' : 158538, 'train' : 67093}

static_RS = 666
np.random.seed(static_RS)
random.seed(static_RS)