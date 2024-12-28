import pandas as pd
import numpy as np
import time
import gc
import pickle

from collections import Counter
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from typing import Any, Dict, Union, List, Optional, Tuple

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import Wav2Vec2Model, Wav2Vec2Config

from comet_ml import Experiment, init
from comet_ml.integration.pytorch import log_model, watch

from pipeline.utils import seed_everything, empty_cache

device = torch.device('cuda' if torch.cuda.is_available()  else 'cpu')


def compute_class_weights_sqrt(y, degree=0.5):
    n_classes = y.nunique()

    weights = len(y) / (n_classes * np.bincount(y).astype(np.float64))
    weights = weights ** degree

    return weights


def get_rare_classes(data, target_letters):
    rare_borders = {}
    for letter in target_letters:
        count_sizes = data[f"{letter}_count"].value_counts(normalize=True).to_dict()
        rare_classes = [i for i, size in count_sizes.items() if size < 0.01]
        if rare_classes:
            rare_borders[letter] = min(rare_classes) - 1
        else:
            rare_borders[letter] = max(count_sizes.keys())

    return rare_borders


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, df, arrays, target_letters):
        self.target_letters = target_letters
        # self.scale_factor = 2**15 # На что делим при переводе аудио из int во float
        df = df.reset_index(drop=True)

        self.files = df['file'].values
        self.arrays = arrays  # [array/self.scale_factor for array in arrays]
        self.texts = df['text'].values
        self.target_letters = target_letters
        for letter in target_letters:
            setattr(self, f"{letter}_count", df[f"{letter}_count"])

        self.labels = df['label'].values

        self.text_lengths = [len(text) for text in df['text']]

    def __len__(self):
        return len(self.arrays)

    def __getitem__(self, idx):

        file = self.files[idx]
        array = self.arrays[idx]
        text = self.texts[idx]
        label = self.labels[idx]  # if self.labels is not None else None

        batch = {
            'file': file,
            'input_values': array,
            'text': text,
            'label': label,
            'text_length': self.text_lengths[idx],
        }

        for letter in self.target_letters:
            batch[f"{letter}_counts"] = getattr(self, f"{letter}_count")[idx]

        return batch


class DataCollator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.max_length = cfg.max_length

    def pad_arrays(self, arrays):
        max_batch_length = max(len(array) for array in arrays)

        arrays = torch.stack([torch.cat([array, torch.zeros(max_batch_length - len(array))]) for array in arrays])
        return arrays

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        arrays = [torch.tensor(feature["input_values"][:self.max_length], dtype=torch.float32) for feature in features]
        arrays = self.pad_arrays(arrays)

        labels = [feature["label"] for feature in features]

        batch = {'input_values': arrays}

        batch["labels"] = torch.tensor(labels, dtype=torch.long) if not np.isnan(labels[0]) else None

        for letter in self.cfg.target_letters:
            batch[f"{letter}_counts"] = torch.tensor([feature[f"{letter}_counts"] for feature in features],
                                                     dtype=torch.long)

        batch["files"] = [feature["file"] for feature in features]

        return batch


class DisordersDetector(nn.Module):

    def __init__(self, cfg, stage):
        super().__init__()
        self.cfg = cfg
        if stage == "pretrain":
            self.backbone = Wav2Vec2Model.from_pretrained(cfg.model_name)
        else:
            # Не подгружаем модель с HF, ведь мы подгрузим претрейн модель
            model_cfg = Wav2Vec2Config.from_pretrained(cfg.model_name)
            self.backbone = Wav2Vec2Model(model_cfg)

        self.stage = stage

        dropout = cfg.dropout
        hidden_dim = 1024
        head_dim = cfg.head_dim

        if stage == "pretrain":
            self.letter_count_heads = nn.ModuleDict({
                f"{letter}_count_head": nn.Sequential(
                    nn.Linear(hidden_dim, head_dim),
                    nn.Dropout(dropout),
                    nn.Linear(head_dim, cfg.letters_num_classes[letter])
                )
                for letter in cfg.target_letters
            })

            # for letter in cfg.target_letters:
            #     num_classes = cfg.letters_num_classes[letter]
            #     setattr(self, f"{letter}_count_head", nn.Sequential(nn.Linear(hidden_size, head_dim),
            #                                     nn.Dropout(dropout),
            #                                     nn.Linear(head_dim, num_classes)))
        else:
            self.disorders_head = nn.Sequential(nn.Linear(hidden_dim, head_dim),
                                                nn.Dropout(dropout),
                                                nn.Linear(head_dim, len(cfg.disorders_class_weights)))

    def forward(self, x):

        hidden_state = self.backbone(x).last_hidden_state
        pooled_output = torch.mean(hidden_state, dim=1)

        output = {}

        if self.stage == 'pretrain':
            # for letter in self.cfg.target_letters:
            #     head = getattr(self, f"{letter}_count_head")
            #     output[f'{letter}_count_output'] = self.g_count_head(pooled_output)
            for letter_head, head in self.letter_count_heads.items():
                output[letter_head[0]] = head(pooled_output)

        else:

            output['disorders'] = self.disorders_head(pooled_output)

        return output

    def freeze_feature_extractor(self):
        self.backbone.feature_extractor._freeze_parameters()

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

        if self.cfg.model_type == "wav2vec":
            self.freeze_feature_extractor()


def get_metric_pretrain(all_predictions, all_targets, is_val=False):
    metrics = []

    for letter in all_predictions.keys():
        # print(all_predictions, letter)
        if len(all_predictions[letter]) == 0: continue
        predictions = np.array(all_predictions[letter]).argmax(axis=-1)
        targets = np.array(all_targets[letter])

        metric = f1_score(targets, predictions, average='macro')
        metrics.append(metric)
        # metrics['letter'] = letter
        print(f"{letter} metric\n", metric)
        if is_val:
            print(confusion_matrix(targets, predictions))

    return metrics, np.mean(metrics)


def get_metric_train(predictions, targets):
    predictions = np.array(predictions['disorders']).argmax(axis=-1)
    targets = np.array(targets['disorders'])

    metric = f1_score(targets, predictions, average='macro')

    print(confusion_matrix(targets, predictions))

    return metric


def model_step(model, stage, batch, cfg, predictions, targets, all_files, criterions):
    loss = 0

    if stage == 'pretrain':

        output = model(batch['input_values'].to(model.backbone.device))

        for letter in cfg.target_letters:
            loss += criterions[f"{letter}_count_head"](output[letter], batch[f'{letter}_counts'].to(model.backbone.device))

            predictions[f'{letter}_count'].extend(F.softmax(output[letter], dim=-1).detach().cpu().numpy())
            targets[f'{letter}_count'].extend(batch[f'{letter}_counts'].cpu().numpy().flatten())

        all_files.extend(np.array(batch['files']))

    else:

        batch_targets = batch['labels'].to(model.backbone.device)

        output = model(batch['input_values'].to(model.backbone.device))

        loss += criterions['disorders'](output['disorders'], batch_targets)

        predictions['disorders'].extend(F.softmax(output['disorders'], dim=-1).detach().cpu().numpy())
        targets['disorders'].extend(batch_targets.cpu().numpy().flatten())

        all_files.extend(np.array(batch['files']))


    return loss


def train_model(model, cfg, dataloader_train, dataloader_val, experiment,
                stage='pretrain', sanity_checking=False):

    if stage == 'pretrain':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr_pretrain)
        num_epochs = cfg.num_epochs_pretrain
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr_train)
        num_epochs = cfg.num_epochs_train

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    best_metric = -1
    best_epoch = -1

    if cfg.linear_probing_frac > 0.0:
        # Замораживаем тело в начале обучения
        model.freeze_backbone()

    num_batches = len(dataloader_train)
    unfreeze_backbone_batch = int(num_batches * min(cfg.linear_probing_frac, 1.0))

    criterions = {}
    if stage == 'pretrain':
        save_weights_name = f"{cfg.save_model_name}-pretrain"
        for letter in cfg.target_letters:
            criterions[f"{letter}_count_head"] = torch.nn.CrossEntropyLoss(weight=cfg.letter_count_weights[letter],
                                                                           label_smoothing=cfg.label_smoothing_pretrain)

    elif stage == 'train':

        save_weights_name = f"{cfg.save_model_name}-train"

        criterions['disorders'] = torch.nn.CrossEntropyLoss(weight=cfg.disorders_class_weights,
                                                            label_smoothing=cfg.label_smoothing_train, reduction='mean')

    for epoch in range(num_epochs):
        if stage == "pretrain" and epoch > best_epoch + cfg.early_stopping_pretrain:  # Метрика не улучшается долго
            break
        elif stage == "train" and epoch > best_epoch + cfg.early_stopping_train:  # Метрика не улучшается долго
            break

        print('*' * 50)
        print('EPOCH', epoch)
        print('*' * 50)

        model.train()

        predictions = {'disorders': []}
        targets = {'disorders': []}

        for letter in cfg.target_letters:
            predictions[f'{letter}_count'] = []
            targets[f'{letter}_count'] = []

        all_files = []

        for batch_idx, batch in enumerate(tqdm(dataloader_train, total=len(dataloader_train), desc='Training')):

            if sanity_checking and batch_idx >= 2: break
            if epoch == 0 and cfg.linear_probing_frac > 0.0 and batch_idx == unfreeze_backbone_batch:
                model.unfreeze_backbone()

            optimizer.zero_grad()

            loss = model_step(model, stage, batch, cfg, predictions,
                              targets, all_files,
                              criterions)

            # experiment.log_metric("loss", loss, step=batch_idx + (num_batches * epoch))

            del batch
            torch.cuda.empty_cache()

            loss.backward()
            optimizer.step()

            if batch_idx > 0 and batch_idx % max(1,
                                                 len(dataloader_train) // cfg.metric_computation_times_per_epoch_train) == 0:

                if stage == 'pretrain':
                    metrics, mean_metric = get_metric_pretrain(predictions, targets, is_val=False)
                    # for letter, metric in zip(cfg.target_letters, metrics):
                        # experiment.log_metric(f"f1_{letter}_count", metric, step=batch_idx + (num_batches * epoch))

                    # experiment.log_metric("f1_mean", mean_metric, step=batch_idx + (num_batches * epoch))
                    metric = mean_metric
                else:
                    metric = get_metric_train(predictions, targets)
                    # experiment.log_metric("f1_disorders", metric, step=batch_idx + (num_batches * epoch))

                print(metric)

            if (batch_idx == 0 and epoch == 0):
                # metric = evaluate(model, cfg, dataloader_val, criterions, experiment,
                #                   stage=stage, is_beggining=True,
                #                   step_idx=batch_idx + (num_batches * epoch))
                model.train()

            elif (batch_idx + 1) % (len(dataloader_train) // cfg.metric_computation_times_per_epoch_val) == 0:
                # metric = evaluate(model, cfg, dataloader_val, criterions, experiment,
                #                   stage=stage,
                #                   step_idx=batch_idx + (num_batches * epoch))
                model.train()

                if metric > best_metric:
                    best_metric = metric
                    best_epoch = epoch
                    torch.save(model.state_dict(), f'{cfg.weights_folder}/{save_weights_name}.pt')

        if sanity_checking:
            break

        scheduler.step()


def evaluate(model, cfg, dataloader_val, criterions,  experiment,
             is_beggining=False,
             limit_num_batches=-1, stage='pretrain',
             step_idx=0, sanity_checking=False):
    num_batches = len(dataloader_val)
    zero_epoch_evaluation_batches = np.ceil(num_batches * min(cfg.zero_epoch_evaluation_frac, 1.0))

    model.eval()

    predictions = {'disorders': []}
    targets = {'disorders': []}

    for letter in cfg.target_letters:
        predictions[f'{letter}_count'] = []
        targets[f'{letter}_count'] = []

    all_files = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader_val):
            if sanity_checking and batch_idx == 5: break
            if batch_idx == limit_num_batches: break  # Оценивал по 50 батчам, т.к. иначе времени бы много занимало, можно исправить и оценивать по фулл даталоадеру, но раз в эпоху
            if is_beggining and batch_idx == zero_epoch_evaluation_batches: break

            loss = model_step(model, stage, batch, cfg,
                              predictions, targets,
                              all_files, criterions)

            del batch
            torch.cuda.empty_cache()

    print('\n VALIDATION')
    print('=*' * 50)

    if stage == 'pretrain':
        metrics, mean_metric = get_metric_pretrain(predictions, targets, is_val=True)

        # for letter, metric in zip(cfg.target_letters, metrics):
            # experiment.log_metric(f"val_f1_{letter}_count", metric, step=step_idx)
            # pass

        # experiment.log_metric("val_f1_mean", mean_metric, step=step_idx)

        metric = mean_metric
    else:
        metric = get_metric_train(predictions, targets)
        # experiment.log_metric("val_f1_disorders", metric, step=step_idx)

    print(metric)
    print('=*' * 50)

    return metric