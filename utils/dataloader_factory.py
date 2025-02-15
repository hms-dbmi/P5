import os
import random
import shutil
import math

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler






def create_dataloader(index, dataset, cfg, result_dir, protein_csv):
    if cfg.Data.dataset_name in ['BagDataset']:
        return create_bag_dataloader(index, dataset, cfg, result_dir,protein_csv)
    else:
        raise NotImplementedError

def create_bag_dataloader_ex(index, dataset_name, cfg, result_dir, label):
    df = pd.read_csv(cfg.Data.external_dir + '/' + label + '.csv')
    df.rename(columns={'case_id': 'patient_id'}, inplace=True)
    df.rename(columns={'slide_id': 'case_id'}, inplace=True)
    print(df.head())

    if cfg.Data.dataset_name == 'BagDataset':
        from datasets.BagDataset import BagDataset
        dataset = BagDataset(df, **cfg.Data)
    else:
        raise NotImplementedError



    if dataset_name == 'train_set':
        if cfg.Train.balance:
            weights = dataset.get_balance_weight()
            dataloader = DataLoader(dataset, batch_size=None, sampler=WeightedRandomSampler(weights, len(weights)),
                                    num_workers=cfg.Train.num_worker)
        else:
            dataloader = DataLoader(dataset, batch_size=None, shuffle=True,
                                    num_workers=cfg.Train.num_worker)
    else:
        dataloader = DataLoader(dataset, batch_size=None, shuffle=False,
                                    num_workers=cfg.Train.num_worker)
    return dataloader

def create_bag_dataloader_infer(index, dataset_name, cfg, result_dir, protein_csv):
    df = pd.read_csv(protein_csv)
    df = df.loc[df['fold'] == index]
    df.rename(columns={'case_id': 'patient_id'}, inplace=True)
    df.rename(columns={'slide_id': 'case_id'}, inplace=True)
    print(df.head())



    if cfg.Data.dataset_name == 'BagDataset':
        from datasets.BagDataset import BagDataset
        dataset = BagDataset(df, **cfg.Data)
    else:
        raise NotImplementedError



    if dataset_name == 'train_set':
        if cfg.Train.balance:
            weights = dataset.get_balance_weight()
            dataloader = DataLoader(dataset, batch_size=None, sampler=WeightedRandomSampler(weights, len(weights)),
                                    num_workers=cfg.Train.num_worker)
        else:
            dataloader = DataLoader(dataset, batch_size=None, shuffle=True,
                                    num_workers=cfg.Train.num_worker)
    else:
        dataloader = DataLoader(dataset, batch_size=None, shuffle=False,
                                    num_workers=cfg.Train.num_worker)
    return dataloader

def create_bag_dataloader(index, dataset_name, cfg, result_dir, protein_csv):
    if dataset_name in ['train_set', 'val_set']:
        if dataset_name == 'train_set':
            df = []
            for i in range(cfg.General.fold_num):
                if i != index:
                    split = pd.read_csv(protein_csv)
                    df.append(split.loc[split['fold'] == i])
            df = pd.concat(df, axis=0)

        else:
            df = pd.read_csv(protein_csv)
            df = df.loc[df['fold'] == index]

        """rename the column names to fit the BagDataset"""
        protein_name = os.path.basename(protein_csv)[:-4]
        df.rename(columns={'case_id': 'case_id_'}, inplace=True)
        df.rename(columns={'slide_id': 'case_id'}, inplace=True)

    if cfg.Data.dataset_name == 'BagDataset':
        from datasets.BagDataset import BagDataset
        dataset = BagDataset(df, **cfg.Data)
    else:
        raise NotImplementedError


    if dataset_name == 'train_set':
        if cfg.Train.balance:
            weights = dataset.get_balance_weight()
            dataloader = DataLoader(dataset, batch_size=None, sampler=WeightedRandomSampler(weights, len(weights)),
                                    num_workers=cfg.Train.num_worker)
        else:
            dataloader = DataLoader(dataset, batch_size=None, shuffle=True,
                                    num_workers=cfg.Train.num_worker, pin_memory=True)
    else:
        dataloader = DataLoader(dataset, batch_size=None, shuffle=False,
                                    num_workers=cfg.Train.num_worker, pin_memory=True)

    return dataloader
