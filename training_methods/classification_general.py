import os

import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score





def validation(index, epoch, model, loader, result_dir, early_stopping, writer, cfg):
    n_classes = cfg.Data.n_classes
    early_stopping_type = cfg.Train.Early_stopping.type

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    val_loss = 0
    model.to(device)
    probs = []
    labels = []
    with torch.no_grad():
        with tqdm(total=len(loader), desc='validate epoch:{}'.format(epoch)) as bar:
            for idx, batch in enumerate(loader):

                x, y = batch['x'].to(device, dtype=torch.float32), \
                       batch['y'].to(device, dtype=torch.long)
                result = model(x)
                logits = result[cfg.Model.logits_field]
                y_prob = torch.softmax(logits, dim=-1)
                loss = loss_fn(logits, y)
                probs.append(y_prob)
                labels.append(y)

                val_loss += loss.item()
                bar.update(1)

    val_loss /= len(loader)

    labels = torch.cat(labels, dim=0).cpu().numpy()
    probs = torch.cat(probs, dim=0).cpu().numpy()
    preds = torch.argmax(torch.from_numpy(probs), dim=1) 
    acc = accuracy_score(labels, np.argmax(probs, axis=-1).astype(np.int32))

    if n_classes == 2:
        auc = roc_auc_score(labels, probs[:, 1])
        f1 = f1_score(labels, preds)

        precision = precision_score(labels, preds)
        recall = recall_score(labels, preds)
        acc_my = accuracy_score(labels, preds)
        specificity = recall_score(1 - labels, 1 - preds)
    else:
        auc = 0
        for i in range(n_classes):
            auc += roc_auc_score((labels == i).astype(np.int), probs[:, i])
        auc /= n_classes

    print('\nVal Set, val_loss: {:.4f}, acc {:.4f}, auc: {:.4f}, f1: {:.4f}, pre: {:.4f}, recall: {:.4f}, '
          'acc_my: {:.4f}, spec: {:.4f}'.format(val_loss, acc, auc, f1, precision, recall, acc_my, specificity))


    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/acc', acc, epoch)

    if early_stopping:
        os.makedirs(os.path.join(result_dir, 'best_checkpoints'), exist_ok=True)
        ckpt_name = os.path.join(result_dir, 'best_checkpoints', "s_{}_checkpoint.pt".format(index))
        if early_stopping_type == 'loss':
            early_stopping(epoch, val_loss, model, ckpt_name=ckpt_name)
        elif early_stopping_type == 'acc':
            early_stopping(epoch, acc, model, ckpt_name=ckpt_name)
        elif early_stopping_type == 'auc':
            early_stopping(epoch, auc, model, ckpt_name=ckpt_name)
        else:
            raise NotImplementedError
        if early_stopping.early_stop:
            print('Early stopping')
            return True

    return False



def evaluation(index, model, loader, result_dir, cfg):
    n_classes = cfg.Data.n_classes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    val_loss = 0
    model.to(device)
    probs = []
    labels = []
    with torch.no_grad():
        with tqdm(total=len(loader)) as bar:
            for idx, batch in enumerate(loader):
                x, y = batch['x'].to(device, dtype=torch.float32), \
                       batch['y'].to(device, dtype=torch.long)
                result = model(x)
                logits = result[cfg.Model.logits_field]
                y_prob = torch.softmax(logits, dim=-1)
                loss = loss_fn(logits, y)

                probs.append(y_prob)
                labels.append(y)

                val_loss += loss.item()
                bar.update(1)

    labels = torch.cat(labels, dim=0).cpu().numpy()
    probs = torch.cat(probs, dim=0).cpu().numpy()
    preds = torch.argmax(torch.from_numpy(probs), dim=1)  
    id_list = loader.dataset.get_id_list()

    df_dict = {'id': id_list, 'label': labels}
    for i in range(n_classes):
        df_dict[f'prob_{i}'] = probs[:, i]

    save_path = os.path.join(result_dir, 'evaluation')
    
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    pd.DataFrame(df_dict).to_csv(os.path.join(save_path, f'preds_{index}.csv'), index=False)

    if n_classes == 2:
        auc = roc_auc_score(labels, probs[:, 1])
        f1 = f1_score(labels, preds)
        precision = precision_score(labels, preds)
        recall = recall_score(labels, preds)
        acc = accuracy_score(labels, preds)
        specificity = recall_score(1 - labels, 1 - preds)

    data = {
        'auc': auc,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'acc': acc,
        'specificity': specificity
    }




    data = pd.DataFrame(data, index=[0])
    data.to_csv(os.path.join(save_path, f'metric_{index}.csv'))


    

def evaluation_external(index, model, loader, result_dir, cfg):
    n_classes = cfg.Data.n_classes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    val_loss = 0
    model.to(device)
    probs = []
    labels = []
    with torch.no_grad():
        with tqdm(total=len(loader)) as bar:
            for idx, batch in enumerate(loader):
                x, y = batch['x'].to(device, dtype=torch.float32), \
                       batch['y'].to(device, dtype=torch.long)
                result = model(x)
                logits = result[cfg.Model.logits_field]
                y_prob = torch.softmax(logits, dim=-1)
                loss = loss_fn(logits, y)

                probs.append(y_prob)
                labels.append(y)

                val_loss += loss.item()
                bar.update(1)

    labels = torch.cat(labels, dim=0).cpu().numpy()
    probs = torch.cat(probs, dim=0).cpu().numpy()
    preds = torch.argmax(torch.from_numpy(probs), dim=1)  
    id_list = loader.dataset.get_id_list()

    df_dict = {'id': id_list, 'label': labels}
    for i in range(n_classes):
        df_dict[f'prob_{i}'] = probs[:, i]

    save_path = os.path.join(result_dir) ## the only difference

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    pd.DataFrame(df_dict).to_csv(os.path.join(save_path, f'preds_{index}.csv'), index=False)

    if n_classes == 2:
        auc = roc_auc_score(labels, probs[:, 1])
        f1 = f1_score(labels, preds)
        precision = precision_score(labels, preds)
        recall = recall_score(labels, preds)
        acc = accuracy_score(labels, preds)
        specificity = recall_score(1 - labels, 1 - preds)


    data = {
        'auc': auc,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'acc': acc,
        'specificity': specificity
    }




    data = pd.DataFrame(data, index=[0])
    data.to_csv(os.path.join(save_path, f'metric_{index}.csv'))