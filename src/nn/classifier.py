import os
from collections import OrderedDict
import logging

from scipy.stats.mstats import gmean
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

import nets
from . import losses as losses_utils
from utils import utils
from . import tools
from .initer import Initer
log = logging.getLogger(__name__)


class Clf(Initer):
    def __init__(self, config, payload):
        super().__init__(config, payload)

    def _validate_epoch(self):
        self.net.eval()
        losses = {"loss": tools.AverageMeter(), "additional": tools.AverageMeter()}
        accuracies = tools.AverageMeter()
        batch_size = self.payload['train_loader'].batch_size
        with tqdm(total=len(self.payload['valid_loader']), desc="Validating", leave=False) as pbar:
            for ind, loader_dict in enumerate(self.payload['valid_loader']):
                with torch.no_grad():
                    loss, acc = self._batch_train_validation(loader_dict)
                for k in losses.keys():
                    losses[k].update(loss[k].item(), batch_size)
                accuracies.update(acc, batch_size)
                pbar.update(1)
        return {k: l.avg for k, l in losses.items()}, accuracies.avg

    def _train_epoch(self, epoch_id):
        self.net.train()
        losses = {"loss": tools.AverageMeter(), "additional": tools.AverageMeter()}
        accuracies = tools.AverageMeter()
        batch_size = self.payload['train_loader'].batch_size
        with tqdm(total=len(self.payload['train_loader']),
                  desc="Epochs {}/{}".format(epoch_id + 1, self.config['n_epochs']),
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{remaining}{postfix}]'
                  ) as pbar:

            for ind, loader_dict in enumerate(self.payload['train_loader']):
                loss, acc = self._batch_train_validation(loader_dict)

                self.optimizer.zero_grad()
                loss['loss'].backward()
                self.optimizer.step()

                for k in losses.keys():
                    losses[k].update(loss[k].item(), batch_size)
                accuracies.update(acc, batch_size)
                loss_str = f"{loss['loss'].item():.5f} | {loss['additional'].item():.5f}"
                pbar.set_postfix(OrderedDict(loss=loss_str,
                                             acc='{0:1.5f}'.format(acc)))
                pbar.update(1)
        return {k: l.avg for k, l in losses.items()}, accuracies.avg

    def _batch_train_validation(self, loader_dict):
        images = loader_dict['img'].type(torch.FloatTensor).to(self.config['device'])
        target = loader_dict['target'].type(torch.FloatTensor).to(self.config['device'])

        probs = self.net.forward(images)
        loss = self._criterion(probs, target)
        acc = self._calculate_acc(target, probs)
        return loss, acc

    def train(self):
        self.net.train()
        log.info(f"Training on {len(self.payload['train_loader'].dataset)} samples.")
        log.info(f"Validating on {len(self.payload['valid_loader'].dataset)} samples.")

        log_df_dict = {"weight": [],
                       "train_loss": [],
                       "train_loss_additional": [],
                       "valid_loss": [],
                       "valid_loss_additional": [],
                       "train_acc": [],
                       "valid_acc": [],
                       "lr": []}

        for epoch_id in range(self.start_epoch, self.config['n_epochs']):
            current_lr = self.optimizer.param_groups[0]['lr']
            log.info(f"Epoch {epoch_id} / {self.config['n_epochs']}, LR = {current_lr}")
            if (epoch_id > 0):
                if self.config['mode_train']['lr_scheduler'] == "plateau":
                    if (current_lr <= 2 * self.config['mode_train']['lowest_lr']):
                        log.info(f'Early stopping with lr = {current_lr}')
                        break

            if epoch_id == self.config['mode_train']['unfreeze']:
                log.info(f"Unfreezing net on {epoch_id} epoch.")
                self.net.module.unfreeze()

            if epoch_id >= 3:
                self.net.module.reset()

            train_loss, train_acc = self._train_epoch(epoch_id)
            valid_loss, valid_acc = self._validate_epoch()
            strs = [f"tl={train_loss['loss']:.5f}",
                    f"tl_a={train_loss['additional']:.5f}",
                    f"vl={valid_loss['loss']:.5f}",
                    f"vl_a={valid_loss['additional']:.5f}",
                    f"ta={train_acc:.3f}",
                    f"va={valid_acc:.3f}"]
            log.info("; ".join(strs) + f"; {self.config['net_name']} {self.payload['fold_num']}\n")

            self.save_state(epoch=epoch_id)

            log_df_dict['train_loss'].append(train_loss['loss'])
            log_df_dict['train_loss_additional'].append(train_loss['additional'])
            log_df_dict['valid_loss'].append(valid_loss['loss'])
            log_df_dict['valid_loss_additional'].append(valid_loss['additional'])
            log_df_dict['train_acc'].append(train_acc)
            log_df_dict['valid_acc'].append(valid_acc)
            log_df_dict['lr'].append(current_lr)
            log_df_dict['weight'].append(f"{self.payload['fold_num']}_{epoch_id}.pth")
            self.training_log_info = pd.DataFrame(log_df_dict)
            self.training_log_info.sort_values('valid_loss', inplace=True)
            csv_path = os.path.join(self.config['out_folder'], f"trainig_log{self.payload['fold_num']}.csv")
            self.training_log_info.to_csv(csv_path, index=False)

            self.lr_scheduler.step(valid_loss['loss'], epoch_id)
            self._load_prev_best_weight(current_lr=current_lr)

    def predict(self):
        self.net.eval()
        self.predict_tta('test')
        self.predict_tta('val')

    def predict_single_loader(self, mode, loader_num, loader):
        ids = []
        probas_batch_list = [None] * len(loader)
        with tqdm(total=len(loader), desc="Predicting") as pbar:
            for ind, loader_dict in enumerate(loader):
                images = loader_dict['img'].type(torch.FloatTensor).to(self.config['device'])
                with torch.no_grad():
                    pred_images = self.net(images)

                probas_batch_list[ind] = self._postprocess_probs(pred_images)
                ids = ids + loader_dict['id']
                pbar.update(1)
        probas = np.concatenate(probas_batch_list, axis=0)

        if self.config['competition_type'] in ['binary', 'multiclass', 'multilabel']:
            if self.config['competition_type'] == 'binary':
                predictions = np.empty([len(loader.dataset)])
            else:
                predictions = np.empty([len(loader.dataset), self.config['num_classes']])
            id_order = np.argsort(ids)
            for i, id in enumerate(id_order):
                if self.config['competition_type'] == 'binary':
                    predictions[i] = probas[id]
                else:
                    predictions[i, :] = probas[id, :]
            self.dump_data(ids=np.sort(ids),
                           preds=predictions,
                           path=os.path.join(self.config['predictions_folder'], f"{mode}_aug_{loader_num}_fold_{self.payload['fold_num']}.pth"))
        else:
            raise NotImplementedError
        return ids, predictions

    def predict_tta(self, mode):
        if mode == 'val':
            loader_list =  self.payload['valid_loader_oof_list']
        elif mode == 'test':
            loader_list =  self.payload['test_loader_list']

        tta_preds_list = []
        for loader_num, loader in enumerate(loader_list):
            log.info('TTA %s, fold %d, on: %s  (%d / %d)' % (mode,
                                                             self.payload['fold_num'],
                                                             str(self.payload['loader_names'][loader_num]),
                                                             loader_num+1,
                                                             len(loader_list)))
            ids, predictions = self.predict_single_loader(mode=mode, loader_num=loader_num, loader=loader)
            tta_preds_list.append(predictions)

        tta_preds = self._avg_tta(tta_preds_list)
        self.dump_data(ids=ids,
                       preds=tta_preds,
                       path=os.path.join(self.config['predictions_folder'], f"{mode}_prediction_fold_{self.payload['fold_num']}.pth"))
