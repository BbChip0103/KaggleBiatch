import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np
import cv2

from .augs import Auger


class ItemGenerator(data.Dataset):
    def __init__(self, data, config, local_config):
        self.data = data
        self.config = config
        self.local_config = local_config
        self.auger = Auger(config=self.config, local_config=self.local_config)

    def __getitem__(self, img_id):
        value = self.data[img_id]
        img = self.read_item_data(value["img_path"])
        dict_ = {'img': img,
                 'id': img_id,
                 # 'aux': value['aux']
                 }
        aug_input_dict = {"image": dict_['img']}
        if self.local_config["include_target"]:
            dict_['target'] = value['target']
            if self.config['competition_type'] == "segmentation":
                aug_input_dict = {"mask": self.read_item_data(dict_['target'])}

        result = self.auger(input_dict=aug_input_dict)
        dict_['img'] = result['image'] / float(self.config['img_denominator'])

        # competiton specific:
        if self.config['siamese']:
            dict_['img'] = self.siamse_crops(dict_['img'])
        else:
            dict_['img'] = cv2.resize(dict_['img'], tuple(self.config['dataset']['resize']))

        dict_['img'] = self.np2torch(dict_['img'])

        if self.config['competition_type'] == "segmentation":
            dict_['target'] = torch.from_numpy(result['mask'].astype(np.float32))
        return dict_

    def read_item_data(self, path):
        if isinstance(path, list):
            img = []
            for i, p in enumerate(path):
                tmp = self.read_image(p)
                img.append(tmp)
            if i == 0:
                img = img[0]
            else:
                img = np.concatenate(img, axis=2)
        else:
            raise NotImplementedError("Img path should be list")

        return img

    def read_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        elif len(img.shape) != 3:
            raise NotImplementedError(f"Given {img.shape}")
        return img

    def siamse_crops(self, img):
        if  self.config['dataset']['resize'][0] == 256:
            resize_size = (512, 512)
            crop_size = 256
            step = 2
        elif self.config['dataset']['resize'][0] == 512:
            resize_size = (2048, 2048)
            crop_size = 512
            step = 4
        else:
            raise NotImplementedError
        final_image = np.zeros((step ** 2, self.config['dataset']['resize'][0], self.config['dataset']['resize'][1], self.config['competition_num_channels']))
        img = cv2.resize(img, resize_size)
        siamse_list = []
        dims = img.shape[0]
        for i in range(dims // crop_size):
            for j in range(dims // crop_size):
                tmp = img[i * crop_size: (i+1) * crop_size, j * crop_size: (j+1) * crop_size, :]
                if self.config['dataset']['resize']:
                    tmp = cv2.resize(tmp, tuple(self.config['dataset']['resize']))
                final_image[i * step + j, ::] = tmp
        return final_image

    @staticmethod
    def np2torch(img):
        img = img.astype(np.float32)
        if len(img.shape) == 2:
            img = np.expand_dim(img, axis=0)
        elif len(img.shape) == 3:
            img = img.transpose(2, 0, 1)
        elif len(img.shape) == 4:
            img = img.transpose(0, 3, 1, 2)
        else:
            raise NotImplementedError
        img = torch.from_numpy(img)
        return img

    def __len__(self):
        return len(self.data)
