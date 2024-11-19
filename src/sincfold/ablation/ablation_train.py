import os
import random
import numpy as np
import torch as tr
import pandas as pd
import shutil

import torch.nn as nn
from sincfold.dataset import SeqDataset, pad_batch
from torch.utils.data import DataLoader

from sincfold.model import sincfold
from sincfold.ablation.ablation_C1D_C2D import sincfold_C1D_C2D
from sincfold.ablation.ablation_no_ResNet2d import sincfold_no_ResNet2d
from sincfold.ablation.ablation_1ResNet2d import sincfold_1ResNet2d
from sincfold.ablation.ablation_no_ResNet1d import sincfold_no_ResNet1d
from sincfold.ablation.ablation_no_ResNet1d_FF import sincfold_no_ResNet1d_FF


def train_net(ablation, train_file, out_path, num_epoch=20, valid_file=None):
    model = { 'sincfold' : sincfold, 
        'C1D_C2D' : sincfold_C1D_C2D,
        'no_ResNet2d' : sincfold_no_ResNet2d,
        'no_ResNet1d' :sincfold_no_ResNet1d,
        '1ResNet2d' : sincfold_1ResNet2d, 
        'no_ResNet1d_FF' : sincfold_no_ResNet1d_FF}

    # Reproducibility
    tr.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    config={}
    valid_file=None
    nworkers=2
    verbose=True
    config['device'] = 'cuda'

    if verbose:
        print("Working on", out_path)

    if "cache_path" not in config:
        config["cache_path"] = "cache/"

    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    else:
        raise ValueError(f"Output path {out_path} already exists")

    if valid_file is not None:
        train_file = train_file
        valid_file = valid_file
    else:
        data = pd.read_csv(train_file)
        valid_split = config["valid_split"] if "valid_split" in config else 0.1
        train_file = os.path.join(out_path, "train.csv")
        valid_file = os.path.join(out_path, "valid.csv")

        val_data = data.sample(frac = valid_split, random_state=42)
        val_data.to_csv(valid_file, index=False)
        data.drop(val_data.index).to_csv(train_file, index=False)

    batch_size = config["batch_size"] if "batch_size" in config else 4
    train_loader = DataLoader(
        SeqDataset(train_file, training=True, **config),
        batch_size=batch_size,
        shuffle=True,
        num_workers=nworkers,
        collate_fn=pad_batch
    )
    valid_loader = DataLoader(
        SeqDataset(valid_file, **config),
        batch_size=batch_size,
        shuffle=False,
        num_workers=nworkers,
        collate_fn=pad_batch,
    )

    net = model[ablation](train_len=len(train_loader), **config)

    best_f1, patience_counter = -1, 0
    patience = config["patience"] if "patience" in config else 30
    if verbose:
        print("Start training...")
    max_epochs = config["max_epochs"] if "max_epochs" in config else 1000
    logfile = os.path.join(out_path, "train_log.csv")

    for epoch in range(num_epoch):
        train_metrics = net.fit(train_loader)

        val_metrics = net.test(valid_loader)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            tr.save(net.state_dict(), os.path.join(out_path, "weights.pmt"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter > patience:
                break

        if not os.path.exists(logfile):
            with open(logfile, "w") as f:
                msg = ','.join(['epoch']+[f"train_{k}" for k in sorted(train_metrics.keys())]+[f"valid_{k}" for k in sorted(val_metrics.keys())]) + "\n"
                f.write(msg)
                f.flush()
                if verbose:
                    print(msg)

        with open(logfile, "a") as f:
            msg = ','.join([str(epoch)]+[f'{train_metrics[k]:.4f}' for k in sorted(train_metrics.keys())]+[f'{val_metrics[k]:.4f}' for k in sorted(val_metrics.keys())]) + "\n"
            f.write(msg)
            f.flush()
            if verbose:
                print(msg)

    # remove temporal files
    shutil.rmtree(config["cache_path"], ignore_errors=True)

    tmp_file = os.path.join(out_path, "train.csv")
    if os.path.exists(tmp_file):
        os.remove(tmp_file)
    tmp_file = os.path.join(out_path, "valid.csv")
    if os.path.exists(tmp_file):
        os.remove(tmp_file)
    return