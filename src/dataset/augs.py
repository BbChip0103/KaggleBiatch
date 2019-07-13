import sys
import importlib
import logging

import torch
import numpy as np
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose, VerticalFlip
)

from utils import utils


log = logging.getLogger(__name__)


class RotateClock90(RandomRotate90):
    def get_params(self):
        return {'factor': 1}


class RotateCounterClock90(RandomRotate90):
    def get_params(self):
        return {'factor': 3}


class RotateCounterClock180(RandomRotate90):
    def get_params(self):
        return {'factor': 2}


augs_dict = {"RandomRotate90": dict(),
             "RotateClock90": dict(),
             "RotateCounterClock90": dict(),
             "RotateCounterClock180": dict(),
             "Flip": dict(),
             "HorizontalFlip": dict(),
             "VerticalFlip": dict(),
             "Transpose": dict(),
             "IAAAdditiveGaussianNoise": dict(),
             "GaussNoise": dict(),
             "MotionBlur": dict(),
             "MedianBlur": dict(blur_limit=3),
             "Blur": dict(blur_limit=3),
             "OpticalDistortion": dict(p=.3),
             "GridDistortion": dict(p=.1),
             "IAAPiecewiseAffine": dict(p=.3),
             "IAASharpen": dict(),
             "IAAEmboss": dict(),
             "RandomContrast": dict(),
             "RandomBrightness": dict(),
             "HueSaturationValue": dict(p=0.3),
             "ShiftScaleRotate": dict(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=.2),
             "CLAHE": dict(clip_limit=2),
             "Noise_of": dict(transforms=[IAAAdditiveGaussianNoise(),
                                          GaussNoise()], p=0.2),
             "Blur_of": dict(transforms=[MotionBlur(p=.2),
                                         MedianBlur(blur_limit=3, p=.1),
                                         Blur(blur_limit=3, p=.1)], p=0.2),
             "Optical_of": dict(transforms=[OpticalDistortion(p=0.3),
                                            GridDistortion(p=.1),
                                            IAAPiecewiseAffine(p=0.3)], p=0.2),
             "Color_of": dict(transforms=[IAASharpen(),
                                          # CLAHE(clip_limit=2),
                                          IAAEmboss(),
                                          RandomContrast(),
                                          RandomBrightness()], p=0.3)}


def str2class(classname):
    return getattr(sys.modules[__name__], classname)


class Auger(object):
    def __init__(self, config, local_config):
        self.config = config
        self.local_config = local_config
        self.setup_aug()

    def setup_aug(self):
        self.make_aug_flags = True
        augs_list = self.local_config['augs_list']
        if (augs_list[0] is None) & (len(augs_list) == 1):
            self.make_aug_flags = False
        else:
            if self.local_config['aug_p'] == 1:
                assert len(augs_list) == 1, f"Augs list cannot be bigger than 1 for p=1, passed: {augs_list}"
                self.auger = self.generate_aug_class(aug_name=augs_list[0], p=1)
            elif (self.local_config['aug_p'] > 1) or (self.local_config["aug_p"] < 0):
                raise ValueError(f"Aug p cannot have such value: {self.local_config[aug_p]}")
            else:
                self.auger = Compose([self.generate_aug_class(aug_name=k) \
                                      for k in augs_list], p=self.local_config['aug_p'])

    def generate_aug_class(self, aug_name, p=None):
        splitted = aug_name.split("_")
        kwargs = augs_dict[aug_name]
        if p is not None:
            kwargs["p"] = p
        if len(splitted) == 1:
            aug_class = str2class(aug_name)
        elif (splitted[1] == "of") & (len(splitted) == 2):
            aug_class = OneOf
        else:
            raise NotImplementedError
        res = aug_class(**kwargs)
        return res

    def __call__(self, input_dict):
        if self.make_aug_flags:
            input_dict = self.call_aug(input_dict=input_dict)
        return input_dict

    def call_aug(self, input_dict):
        if input_dict.get("mask"):
            raise NotImplementedError("Not implemented aug for mask")
        result = self.auger(**input_dict)
        return result
