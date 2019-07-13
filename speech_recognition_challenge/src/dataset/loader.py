import torch.utils.data as data
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from sklearn import preprocessing
from torch.utils.data import DataLoader
from sklearn.utils import shuffle
import numpy as np
import os
import cv2
import pandas as pd
import sys
from tqdm import tqdm
import torch
import glob
from skimage.io import imread
import skimage.transform
from PIL import Image
import cv2
import joblib
import json

from scipy.io import wavfile
from scipy import signal
sys.path.insert(0,'..')
import config
from img import transformer
import img.augmentation as aug
from utils import utils
from utils import get_smarties

class Dataset(object):
    def __init__(self, RS, proj_folder, silence_binary = False, n_folds = 5, pseudo_file = None, include_double_words = False):
        """
            A tool used to automatically download, check, split and get
            relevant information on the dataset

            pseudo_folder - add test images to train            
        """
        self.n_folds = n_folds
        self.train_ids_list = []
        self.val_ids_list = []
        self.RS = RS
        self.pseudo_file = pseudo_file
        self.proj_folder = proj_folder
        self.silence_binary = silence_binary 
        self.include_double_words = include_double_words

        #
        print('Reading dataframes..')
        self.read_train()
        self.read_test()

        # #splitting
        noise_persons = ['dude_miaowing', 'running_tap', 'exercise_bike', 'doing_the_dishes', 'pink_noise', 'white_noise']
        self.kf = KFold(n_splits = n_folds, random_state = self.RS, shuffle = True)
        unique_person =np.array([x for x in self.train['person_id'].unique().tolist() if x not in noise_persons])
        for  j, (train_indices, val_indices) in enumerate(self.kf.split(unique_person)) : 
            val_noise_persons =  [noise_persons[j]]
            if j == 0 :
                train_noise_persons = noise_persons[j+1:]
            elif j == n_folds - 1:
                train_noise_persons = noise_persons[:j]
                val_noise_persons =  noise_persons[j:]
            else:
                train_noise_persons = noise_persons[:j] + noise_persons[j+1:]

            train_person = unique_person[train_indices].tolist() + train_noise_persons
            val_person = unique_person[val_indices].tolist() + val_noise_persons

            self.train_ids_list.append(self.train[self.train['person_id'].isin(train_person)][config.id_col].tolist())
            self.val_ids_list.append(self.train[self.train['person_id'].isin(val_person)][config.id_col].tolist())

            print('train',train_noise_persons )
            print('val',val_noise_persons )

        if self.pseudo_file is not None:
            self.pseudo_file = config.OUTPUT_FOLDER + self.pseudo_file
            stratified = StratifiedKFold(n_splits = n_folds, random_state = self.RS, shuffle = True)
            print('\nUsing pseudo from %s\n'%self.pseudo_file)
            self.train_pseudo = pd.read_csv(self.pseudo_file)
            self.train = pd.concat([self.train, self.train_pseudo])
            self.train.reset_index(inplace = True, drop = True)
            #add pseudo only to train in kfold manner
            for i, (pseudo_train_indices, pseudo_val_indices) in enumerate(self.kf.split(self.train_pseudo , self.train_pseudo[config.target_col] )) : 
                pseudo_train_ids = self.train_pseudo.loc[pseudo_train_indices, :][config.id_col].tolist()
                l = len(self.train_ids_list[i])
                self.train_ids_list[i] += pseudo_train_ids
                print('Fold %d before pseudo had %d, now %d'%(i, l, len(self.train_ids_list[i])))

        ## depricated label encoder
        # self.le = preprocessing.LabelEncoder()
        # self.train[config.target_col] = self.le.fit_transform(self.train[config.target_col].tolist())
        # joblib.dump(self.le, self.proj_folder + 'label_encoder.dump')

        if self.silence_binary:
            self.train[config.target_col] = self.train[config.target_col].apply(lambda x: 1 if x == 'silence' else  0)
        else:
            self.train[config.target_col] = self.train[config.target_col].map(config.mapping_dict)

        # self.train = pd.concat([self.train.drop(config.target_col, 1), target_df], axis = 1)

        #save
        # if self.pseudo_file is None:
        #     self.train.to_csv('gt_train.csv', index = False)
        #     self.test.to_csv('gt_test.csv', index = False)

    def read_test(self):
        self.test = None
        for f in ['test/audio/', 'custom/silence/']:
            audio_path = config.DATA_FOLDER + f
            all_files = [y for y in os.listdir(audio_path) if '.wav' in y]
        
            df = pd.DataFrame([{'path' : audio_path + x, config.id_col : x.split('.')[0]} for x in all_files])      
            if self.test is not None:
                self.test = pd.concat([self.test, df])
            else: 
                self.test = df

            self.test.reset_index(drop = True, inplace = True)
        print(self.test['id'].nunique(), len(self.test))

    def read_train(self):
        audio_path = config.DATA_FOLDER +  'train/audio/'

        subFolderList = []
        for x in os.listdir(audio_path):
            if os.path.isdir(audio_path + '/' + x):
                if '_background_noise_' not in x:
                    subFolderList.append(x)
                
        sample_audio = []
        total = 0

        dict_ = {}

        for sub in subFolderList:
            # get all the wave files
            all_files = [y for y in os.listdir(audio_path + sub) if '.wav' in y]
            sample_audio += [{'path': audio_path  + sub + '/'+ x, 
                              config.id_col : x.split('.')[0] + '_' + sub, 
                              'person_id' : x.split('_')[0],
                              config.target_col : self.verify_label(sub)} for x in all_files]
        self.train = pd.DataFrame(sample_audio)

        if self.include_double_words:
            # synthetic 
            # double_words = pd.read_csv(config.DATA_FOLDER + 'custom/double_words.csv')
            # double_words[config.target_col] = "unknown"
            # double_words["path"] = config.DATA_FOLDER + "custom/double_words/" + double_words["id"] + '.wav'
            # self.train = pd.concat([double_words[['path', config.id_col, 'person_id', config.target_col]], self.train])

            # predicted unknown unknowns
            un_un = json.load(open(config.DATA_FOLDER + "unknown_unknown_nn16_nn19_3090.json", "r"))
            unknown_unknowns_df = pd.DataFrame({config.id_col : un_un['id']})
            unknown_unknowns_df[config.target_col] = "unknown"
            unknown_unknowns_df['path'] = config.DATA_FOLDER + "test/audio/" + unknown_unknowns_df[config.id_col] + ".wav"
            unknown_unknowns_df["person_id"] = unknown_unknowns_df[config.id_col]
            self.train = pd.concat([unknown_unknowns_df, self.train])

        # self.train = shuffle(self.train, random_state = config.static_RS)
        self.train.reset_index(inplace = True, drop = True)
        print(self.train[config.id_col].nunique(), self.train['person_id'].nunique(), len(self.train), self.train['target'].nunique())

    @staticmethod
    def verify_label(label):
        # allowed_train_labels = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence']  
        allowed_train_labels = config.categories.keys()
        if label not in allowed_train_labels:
            label = 'unknown'
        return label


class CustomPredictionDataset(object):
    def __init__(self, folder):
        print('Reading dataframes..')
        audio_path = config.DATA_FOLDER +  'custom/%s/'%folder
        all_files = [y for y in os.listdir(audio_path) if '.wav' in y]#[:100]
        self.test = pd.DataFrame([{'path' : audio_path + x, config.id_col : x} for x in all_files])             
    

class ImageDataset(data.Dataset):
    def __init__(self, X_data, include_target, target_cols = None, u = 0.5, X_transform = None):
        self.X_data = X_data
        self.include_target = include_target
        self.target_cols = target_cols
        self.X_transform = X_transform
        self.u = u

    def __getitem__(self, index):
        np.random.seed()
        path = self.X_data.iloc[index]['path'] 
        
        sr, wav = wavfile.read(path)
        img_id = self.X_data.iloc[index][config.id_col]

        # create torch from wav
        wav_torch = np.zeros((1,16000))
        wav_torch[0,:len(wav)] = wav
        wav_torch = wav_torch.astype(np.float32)
        wav_torch = torch.from_numpy(wav_torch)

        if self.X_transform:
            wav = self.X_transform(wav, **{'u' : self.u})

        wav_img = utils.log_specgram(wav)
        img = np.zeros((1,99,161))
        img[0,:wav_img.shape[0], :wav_img.shape[1]] = wav_img
        img_numpy = img.astype(np.float32)
        img_torch = torch.from_numpy(img_numpy)

        dict_ = {'img' : img_torch,
                'id' : img_id, 
                'img_np' : img_numpy,
                'aux' : wav_torch}

        if self.include_target:
            dict_['target'] = self.X_data.iloc[index][config.target_col]

        return dict_

    def __len__(self):
        return len(self.X_data)

if __name__ == '__main__':
    ds = Dataset(15, proj_folder = config.OUTPUT_FOLDER + 'zzz/', 
                    silence_binary = False,
                     pseudo_file = None, include_double_words = True)
    print(ds.train.head())
    print(ds.test.head())
    print(ds.train.info())
    print(ds.train['target'].value_counts())

    # print(ds.train['target'].value_counts())

    # batch_size = 2
    # train_ds = ImageDataset(ds.train, include_target = True,
    #                          X_transform = aug.data_transformer
    #                         )
    # train_loader = DataLoader(train_ds, batch_size,
    #                         sampler = RandomSampler(train_ds),
    #                         num_workers = 5,
    #                         pin_memory= config.USE_CUDA )

    # for i, dict_ in enumerate(train_loader):
    #     for j in np.arange(batch_size):
    #         print(dict_['img'].min(), dict_['img'].max(), dict_['img'].shape)
    #         # print(dict_['target'])
    #         print(dict_['aux'].shape)

    #     if i > 1:
    #         break
