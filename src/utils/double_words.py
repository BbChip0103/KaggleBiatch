import os
import pandas as pd
import glob
from tqdm import tqdm
from scipy.io import wavfile
import numpy as np
from sklearn.utils import shuffle

sys.path.insert(0,'..')
import config

class SingleDoubleWordGenerator(object):
    def __init__(self, path1, target1, path2, target2):
        _, self.wav_read1 = wavfile.read(path1)
        _, self.wav_read2 = wavfile.read(path2)
        self.target1 = target1
        self.target2 = target2
        
        self.delta = 8000
            
    def caclulate_energy_coef(self):
        wav1 = self.wav_read1.copy()
        wav2 = self.wav_read2.copy()
        wav1 = wav1.astype(float)
        wav2 = wav2.astype(float)
        energy1 = float(np.sqrt(wav1.dot(wav1) / float(len(self.target1))))
        energy2 = float(np.sqrt(wav2.dot(wav2) / float(len(self.target2))))
        return (energy2 / energy1)


    def generate_double_word(self):
        divider1 = np.argmax(np.abs(self.wav_read1))
        max_vol1 = np.max(np.abs(self.wav_read1))

        divider2 = np.argmax(np.abs(self.wav_read2))
        max_vol2 = np.max(np.abs(self.wav_read2))

        coef = self.caclulate_energy_coef()
        self.wav_final = np.zeros(16000)

#         print(divider1, divider2)
        len1 = np.clip(np.abs(divider1),0,self.delta)
        self.wav_final[self.delta-len1:self.delta] = coef * self.wav_read1[divider1 - len1 :divider1]

        len2 = np.clip(np.abs(len(self.wav_read2) - divider2),0,self.delta)
        self.wav_final[self.delta:self.delta + len2] = self.wav_read2[divider2 :divider2 + len2]
        vol_adjust = np.mean(np.array([max_vol1, max_vol2]))

        self.wav_final = vol_adjust * (self.wav_final) / np.max(np.abs(self.wav_final))
        self.wav_final = np.asarray( self.wav_final, dtype=np.int16)


    def visualize(self):
        f,ax = plt.subplots(3,1,figsize = (20,12))
        ax[0].plot(np.arange(len(self.wav_read1)), self.wav_read1)
        ax[0].set_title("Original word: %s"%self.target1)
        ax[1].plot(np.arange(len(self.wav_read2)), self.wav_read2)
        ax[1].set_title("Original word: %s"%self.target2)
        ax[2].plot(np.arange(len(self.wav_final)), self.wav_final)
        ax[2].set_title("Double part file")
        
    def __call__(self):
        self.caclulate_energy_coef()
        self.generate_double_word()
        return self.wav_final

class DoubleWordGenerator(object):
    def  __init__(self):
        self.gt_train = pd.read_csv(config.DATA_FOLDER + 'gt_train.csv')
        self.persons = (self.gt_train.groupby(['person_id'])['target'].count() > 10).index.tolist()
        self.save_folder = config.DATA_FOLDER + 'custom/double_words/'
    
    def __call__(self):
        list_ = []
        for p in tqdm(self.persons):
            slice_df = self.gt_train[self.gt_train['person_id'] == p].copy()
            unique_targets = slice_df['target'].unique().tolist()
            for t in unique_targets:
                other_targets = list(unique_targets)
                other_targets.remove(t)
                
                path1 = shuffle(slice_df[slice_df['target'] == t])['path'].values[0]
                for t2 in other_targets:
                    if not t2.startswith(t[:1]):
                        path2 = shuffle(slice_df[slice_df['target'] == t2])['path'].values[0]
                        resulting_name = p + "_" + t + "_" + t2
                        
                        list_.append({'id' : resulting_name, 'person_id':p, 'target1':t, 
                                                  'target2':t2, 'path1': path1, 'path2' : path2})
                        
                        self.wav = SingleDoubleWordGenerator(path1, t, path2, t2)()
                        wavfile.write(self.save_folder + resulting_name + '.wav' , 16000, self.wav)
                        
        self.df = pd.DataFrame(list_)
        self.df.to_csv(config.DATA_FOLDER + 'custom/double_words.csv', index = False)
    
if __name__ == '__main__':
    d = DoubleWordGenerator()()