import logging
import os
import numpy as np
import zipfile
import importlib


def save_src_to_zip(save_path, exclude_folders=[], name='src'):
    goal_zip_folder = os.path.join("../", 'src')
    zf = zipfile.ZipFile(os.path.join(save_path, name + "_zipped.zip"), "w")
    for dirname, subdirs, files in os.walk(goal_zip_folder):
        for ef in exclude_folders:
            if (ef in subdirs):
                subdirs.remove(ef)
        zf.write(dirname)
        for filename in files:
            zf.write(os.path.join(dirname, filename))
    zf.close()


def create_path(path, create=False):
    for i in np.arange(100000):
        new_path = os.path.join(path, str(i))
        if not os.path.exists(new_path):
            break

    if create:
        os.makedirs(new_path)

    return new_path, i


def load_class(full_name):
    module_name, class_name = full_name.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def set_logger(out_folder, name):
    log_path = os.path.join(out_folder, name)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ])
    return logging.getLogger(__name__)
