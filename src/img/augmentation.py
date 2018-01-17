import cv2
import numpy as np
from skimage.transform import rotate
from scipy.io import wavfile
from scipy import signal
import sys, os

sys.path.insert(0,'..')
import config

def random_speed(wav, u = 0.5):
    res = wav
    # print('speed', np.random.random())
    if np.random.random() < u:
        speed_rate = np.random.uniform(0.7,1.3)
        wav_speed_tune = cv2.resize(wav, (1, int(len(wav) * speed_rate))).squeeze()

        if len(wav_speed_tune) < 16000:
            pad_len = 16000 - len(wav_speed_tune)
            wav_speed_tune = np.r_[np.random.uniform(-0.001,0.001,int(pad_len/2)),
                                wav_speed_tune,
                                np.random.uniform(-0.001,0.001,int(np.ceil(pad_len/2)))]
        else: 
            cut_len = len(wav_speed_tune) - 16000
            wav_speed_tune = wav_speed_tune[int(cut_len/2):int(cut_len/2)+16000]
            res = wav_speed_tune
    return res

def random_noise(wav, u = 0.5):
    #https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/discussion/46839
    res = wav
    # print('noiza', np.random.random())
    if np.random.random() < u:
        if len(wav) != 16000:
            new_wav = np.zeros(16000)
            new_wav[:len(wav)] = wav.copy()
            wav = new_wav
       
        i_n = np.random.random_integers(0,len(config.noise_files) - 1)

        samplerate, bg_wav  = wavfile.read(config.noise_files[i_n])

        start_ = np.random.random_integers(bg_wav.shape[0]-16000)
        bg_slice = np.array(bg_wav[start_ : start_+16000]).astype(float)
        wav = wav.astype(float)

        #get energy
        noise_energy = float(np.sqrt(bg_slice.dot(bg_slice) / float(bg_slice.size)))
        data_energy = float(np.sqrt(wav.dot(wav) / float(wav.size)))
        coef = (data_energy / noise_energy)
        wav_with_bg = wav * np.random.uniform(0.8, 1.2) + bg_slice * np.random.uniform(0, 0.4)* coef
        res = wav_with_bg
    return res

def random_shift(wav, u = 0.5):
    res = wav
    # print('shift', np.random.random())
    if np.random.random() < u:
        start_ = int(np.random.uniform(-4800,4800))
        if start_ >= 0:
            wav_time_shift = np.r_[wav[start_:], np.random.uniform(-0.001,0.001, start_)]
        else:
            wav_time_shift = np.r_[np.random.uniform(-0.001,0.001, -start_), wav[:start_]]
        res = wav_time_shift
    return res

def data_transformer(wav, u = 0.5):
    wav = random_shift(wav, u = 0.5)       
    wav = random_speed(wav, u = 0.5)               
    wav = random_noise(wav, u = 0.5)  

    return wav