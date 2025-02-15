

import time
import argparse
import os
import shutil
import torch
from tensorboardX import SummaryWriter

from utils.utils import read_yaml, seed_torch, EarlyStopping
from utils.model_factory import load_model
from utils.dataloader_factory import create_dataloader
from utils.optimizer_factory import create_optimizer
from utils.training_method_factory import create_training_loop, create_validation, create_evaluation
import pandas as pd
from tqdm import tqdm


def training_task(index, result_dir, cfg, protein_csv):
    model = load_model(cfg)
    train_loader = create_dataloader(index, 'train_set', cfg, result_dir, protein_csv)
    val_loader = create_dataloader(index, 'val_set', cfg, result_dir, protein_csv)

    print('#'*30 + '\n' 
        f'load dataset successfully!\n'
        f'train set size: {len(train_loader.dataset)}\n'
        f'val set size: {len(val_loader.dataset)}\n'
        + '#'*30
    )

    # tensorboardX writer
    writer_dir = os.path.join(result_dir, 'log', str(index))
    if not os.path.isdir(writer_dir):
        os.makedirs(writer_dir)
    writer = SummaryWriter(writer_dir, flush_secs=15)

    optimizer = create_optimizer(model, cfg)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.Train.CosineAnnealingLR.T_max,
        eta_min=cfg.Train.CosineAnnealingLR.eta_min,
        last_epoch=-1
    )

    early_stopping = EarlyStopping(
        patience=cfg.Train.Early_stopping.patient,
        stop_epoch=cfg.Train.Early_stopping.stop_epoch,
        type= 'min' if cfg.Train.Early_stopping.type in ['loss'] else 'max'
    )


    current_epoch = 0

    train_loop = create_training_loop(cfg)
    validation = create_validation(cfg)
    evaluation = create_evaluation(cfg)

    print('#'*10 + ' training task start! ' + '#' * 10)


    for epoch in range(current_epoch, cfg.Train.max_epochs):

        lr = scheduler.get_last_lr()[0]
        print('learning rate:{:.8f}'.format(lr))
        writer.add_scalar('train/lr', lr, epoch)

        # training
        train_loop(
            epoch=epoch,
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            writer=writer,
            cfg=cfg
        )

        # validation
        stop = validation(
            index=index,
            epoch=epoch,
            model=model,
            loader=val_loader,
            result_dir=result_dir,
            early_stopping=early_stopping,
            writer=writer,
            cfg=cfg
        )


        if stop:
            break
        scheduler.step()

    # after all epochs training, save the best metrics


    print('===start to save the best metrics!===')
    model.load_state_dict(torch.load(os.path.join(result_dir, 'best_checkpoints', "s_{}_checkpoint.pt".format(index))),
                          strict=False)
    evaluation(
        index=index,
        model=model,
        loader=val_loader,
        result_dir=result_dir,
        cfg=cfg
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/task0_sample.yaml')
    parser.add_argument('--begin', type=int, default=0)
    parser.add_argument('--end', type=int, default=5)
    parser.add_argument('--set_seed', action='store_true')
    args = parser.parse_args()

    cfg = read_yaml(args.config_path)
    csv_path = cfg.Data.split_dir
    protein_list = os.listdir(csv_path)

    print('num_iter', len(protein_list))

    for protein in tqdm(protein_list):
        protein_name = protein[:-4] # remove ".csv"
        print('=====protein_name', protein_name)
        protein_csv = os.path.join(cfg.Data.split_dir, protein)
        

        """judge if the number of positive is less than 10"""
        ## remove label=1 and label=0 < 10 (pid)
        split_file = pd.read_csv(protein_csv)
        df_remove_1 = split_file[split_file[protein_name] == 1]
        df_remove_1 = df_remove_1.drop_duplicates(subset='patient_id')
        df_remove_0 = split_file[split_file[protein_name] == 0]
        df_remove_0 = df_remove_0.drop_duplicates(subset='patient_id')

        if len(df_remove_1) < 10 or len(df_remove_0) < 10: # only for dfci test 5, cptac训练的时候设置为5; other 10
            continue

        result_dir = os.path.join(cfg.General.result_dir, protein_name, args.config_path.split('/')[-1] + f'_seed{cfg.General.seed}')
        os.makedirs(result_dir, exist_ok=True)
        print('result_dir', result_dir)

        if os.path.exists(os.path.join(result_dir, 'evaluation/metric_all.csv')):
            continue

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for i in range(args.begin, args.end):
            if args.set_seed:
                seed_torch(device, cfg.General.seed + i)
            training_task(i, result_dir, cfg, protein_csv)

        # concat these metric csv
        metric_all = pd.DataFrame()
        metric_path = os.path.join(result_dir, 'evaluation')
        for i in range(args.begin, args.end):
            for file in os.listdir(metric_path):
                if file.endswith(str(i)+'.csv') and file.startswith('metric'):
                    data = pd.read_csv(os.path.join(metric_path, file))
                    # metric_all.append(data)
                    metric_all = pd.concat([metric_all, data])
        metric_all.loc[11] = metric_all.apply(lambda x: x.mean())
        metric_all.to_csv(os.path.join(metric_path, 'metric_all.csv'))








