import logging
import os
from functools import reduce

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, fbeta_score

from utils import utils
from .cyclic_lr import CyclicLR
from nets.base import Siamese
log = logging.getLogger(__name__)


class Initer(object):
    def __init__(self, config, payload):
        self.config = config
        self.payload = payload
        self.init_net()
        self.init_optim()
        self.init_scheduler()
        self.load_state()
        self.net.to(self.config['device'])
        if (self.config["mode_train"]['unfreeze'] > 0) & (self.config['pretrained_weights'] is None):
            self.net.module.freeze()

    def init_net(self):
        net = utils.load_class(".".join(["nets", self.config['net_class']]))(config=self.config)
        if self.config['siamese']:
            net = Siamese(one_head=net)
        self.net = nn.DataParallel(net)

    def init_optim(self):
        if self.config['mode_train']['optim'] == "sgd":
            self.optimizer =  optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()),
                                        lr=self.config['mode_train']['start_lr'],
                                        momentum=0.9,
                                        weight_decay=0.0001)
        elif self.config['mode_train']['optim'] == "rms":
            self.optimizer =  optim.RMSprop(filter(lambda p: p.requires_grad, self.net.parameters()),
                                            lr=self.config['mode_train']['start_lr'],
                                            momentum=0.9,
                                            weight_decay=0.0001)
        elif self.config['mode_train']['optim'] == "adam":
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()),
                                        lr=self.config['mode_train']['start_lr'])
        else:
            raise NotImplementedError(f"No such optimizer {self.config['mode_train']['optim']}")

    def init_scheduler(self):
        if self.config['mode_train']['lr_scheduler'] == 'plateau':
            self.lr_scheduler = ReduceLROnPlateau(optimizer=self.optimizer,
                                                  mode='min',
                                                  patience=8,
                                                  verbose=True,
                                                  min_lr=self.config['mode_train']['lowest_lr'])
        elif self.config['mode_train']['lr_scheduler'] == 'cyclic':
            self.lr_scheduler = CyclicLR(optimizer=self.optimizer,
                                         base_lr=self.config['mode_train']['lowest_lr'],
                                         max_lr=self.config['mode_train']['start_lr'],
                                         step_size=2000,
                                         mode='triangular')
        else:
            raise NotImplementedError(f"No such lr scheduler {self.config['train_mode']['lr_scheduler']}")

    def load_state(self):
        self.start_epoch = 0
        if self.config['pretrained_weights']:
            state = torch.load(self.config['pretrained_weights'], map_location='cpu')
            self.net.load_state_dict(state['net'])
            # self.optimizer.load_state_dict(state['optim'])
            # self.lr_scheduler.load_state_dict(state['lr_scheduler'])
            self.start_epoch = state['epoch']

    def save_state(self, epoch):
        weights_path = os.path.join(self.config['weights_folder'],
                                    f"{self.payload['fold_num']}_{epoch}" + ".pth")
        res = {"net": self.net.state_dict(),
               "optim": self.optimizer.state_dict(),
               # "lr_scheduler": self.lr_scheduler.state_dict(),
               "epoch": epoch}
        torch.save(res, weights_path)

    def _criterion(self, probs, labels):
        res = {"additional": torch.Tensor([0])}
        if self.config['competition_type'] == 'binary':
            res['loss'] = nn.BCEWithLogitsLoss().forward(probs, labels)
        elif self.config['competition_type'] == "multiclass":
            res['loss'] = nn.CrossEntropyLoss().forward(probs, labels)
        elif self.config['competition_type'] == "multilabel":
            res = self._criterion_multilabel(probs, labels)
        else:
            raise NotImplementedError
        return res

    def _criterion_multilabel(self, probs, labels):
        if "weighted" in self.config['mode_train']['loss'].keys():
            loss_ = nn.BCEWithLogitsLoss(reduction="none")(probs, labels)
            res['additional'] = loss_.mean()
            norm_coeff =  0.5 / self.payload['class_weights'].median()
            res['loss'] = (self.payload['class_weights'] * loss_).mean() * norm_coeff
        elif "ohnm" in self.config['mode_train']['loss'].keys():
            loss_ = nn.BCEWithLogitsLoss(reduction="none")(probs, labels)
            batch_size = loss_.shape[0]
            res['additional'] = loss_.mean()
            ratio = self.config['mode_train']['loss']['ohnm']['ratio']
            topk_inds = loss_.sum(dim=1).topk(int(batch_size * ratio))[1]
            loss_topk = loss_.index_select(0, topk_inds)
            res['loss'] = loss_topk.mean()
        else:
            res['loss'] = nn.BCEWithLogitsLoss()(probs, labels)
        return res

    def _calculate_acc(self, target, probs):
        target = target.cpu().numpy()
        preds = self._postprocess_probs(probs)
        if self.config['competition_type'] == "binary":
            res = accuracy_score(target, (preds > 0.5) * 1)
        elif self.config['competition_type'] == "multiclass":
            preds = preds.round().argmax(axis=1)
            res = accuracy_score(target, preds)
        elif self.config['competition_type'] == "multilabel":
            res = f1_score(target, (preds > 0.5) * 1, average='macro')
        else:
            raise NotImplementedError
        return res

    def _postprocess_probs(self, probs):
        if self.config['competition_type'] == "multiclass":
            res = nn.Softmax()(probs)
        if self.config['competition_type'] in ["multilabel", "binary"]:
            res = nn.Sigmoid()(probs)
        else:
            raise NotImplementedError(f"No prob function for mode {self.config['competition_type']}")
        res = res.detach().cpu().numpy().astype(np.float32)
        return res

    def _load_prev_best_weight(self, current_lr):
        if self.optimizer.param_groups[0]['lr'] < current_lr:
            best_weight_name = self.training_log_info[self.training_log_info['lr'] == current_lr].sort_values('valid_loss').head(1)['weight'].item()
            log.info(f'Decrease lr - loading best weight {best_weight_name} for lr {current_lr}')
            best_weight = torch.load(os.path.join(self.config['weights_folder'], f'{best_weight_name}'))
            self.net.load_state_dict(best_weight['net'])

    def _avg_tta(self, tta_preds_list):
        log.info("Geomeaning TTA.")
        N = float(len(tta_preds_list))
        res = reduce(lambda x, y: x*y, tta_preds_list)
        res = np.power(res, 1 / N)
        return res

    def dump_data(self, ids, preds, path):
        res = {"id": ids, "pred": torch.from_numpy(preds)}
        torch.save(res, path)

    def remove_unnecessary_weights(self, n=7):
        folder = self.output_folder
        training_info = pd.read_csv(os.path.join(folder, f'trainig_log{self.payload["fold_num"]}.csv'))
        training_info.sort_values('valid_loss', inplace=True)
        top_weights = training_info.head(n)['weight'].tolist()
        unnecessary_weigths = training_info[~training_info['weight'].isin(
            top_weights)]['weight'].tolist()
        log.info(f'Removing unnecessary weights: {unnecessary_weigths}')
        for bad_weight in unnecessary_weigths:
            os.remove(os.path.join(folder, f"fold_{self.payload['fold_num']}", str(bad_weight) + '.pth'))
