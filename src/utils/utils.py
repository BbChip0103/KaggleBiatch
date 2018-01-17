import sys, os
from scipy import signal
import numpy as np
from scipy.io import wavfile
import zipfile

def create_unique_path(path, create = False):
    r"""Checks if existing path doesn't exist - and if exist 
    add new path (so old path isn't rewritten). It works for folders and files
    -------
    Args:
    - absolute path to file / folder
    - create - whether or note to create folder (not applicable to files)
    Note: if folder it doesn't matter whether there is "/" or not at the end of path
    -------
    Returns:
    - returns new path
    """
    #-1 folder without "/", 0 - folder with "/", 1 - file extension
    path_file = -1
    file_extension = ''
    
    if path.endswith('/'): 
        path_file = 0
        path= path[:-1]
    elif '.' in path: 
        path_file = 1
        [path, file_extension] = path.split('.')
        file_extension = '.' + file_extension

    new_path = path

    for i in np.arange(1000):
        if os.path.exists(new_path + file_extension):
            new_path = path + '_' + str(i)
        else: break

    if path_file==0:
        new_path+= '/'
    elif path_file == 1:
        new_path += file_extension
    
    if create and (path_file==0 or path_file==-1):
        os.makedirs(new_path)

    return new_path


def zip_src_folder(save_path, exclude_folders = [], goal_dir = '.', name = 'src'):
    r"""Zips whole src folder and saves it to save_path.
    -------
    Args:
    - save_path = where to save .zip file
    - exclude_folders = list of folders to be excluded
    - goal_dir = if zip all files in current folder = ".", if in parent folder = "../"
            so if you use this function in main, then goal_dir = "."
    - name = name of .zip file
    Note: by default put this function in file that is in src/ folder (i.e put it in main.py).
    """
    zf = zipfile.ZipFile(save_path + name + "_zipped.zip", "w")
    for dirname, subdirs, files in os.walk(goal_dir):
        for ef in exclude_folders:
            if (ef in subdirs):
                subdirs.remove(ef)
        zf.write(dirname)
        for filename in files:
            zf.write(os.path.join(dirname, filename))
    zf.close()


def save_src_to_zip(save_path, exclude_folders = [], name = 'src'):
    goal_dir = "../"
    zf = zipfile.ZipFile(save_path + name + "_zipped.zip", "w")
    for dirname, subdirs, files in os.walk(goal_dir + 'src/'):
        for ef in exclude_folders:
            if (ef in subdirs):
                subdirs.remove(ef)
        zf.write(dirname)
        for filename in files:
            zf.write(os.path.join(dirname, filename))
    zf.close()

def read_list_from_file(filename):
    """Reads list of strings from file"""
    f = open(filename, 'r')
    list_ = f.read().splitlines()
    f.close()
    list_ = list(set(list_))
    return list_


def log_specgram(audio, sample_rate = 16000, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, _, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return np.log(spec.T.astype(np.float32) + eps)

def read_wav(img_path):
    sample_rate, audio  = wavfile.read(img_path)
    res = log_specgram(audio, sample_rate)
    return res

