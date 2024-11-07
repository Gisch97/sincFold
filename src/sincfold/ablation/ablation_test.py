import os
import random
import numpy as np
import torch as tr
import pandas as pd
import shutil


from torch.utils.data import DataLoader
from sincfold.dataset import SeqDataset, pad_batch
from sincfold.model import sincfold
from sincfold.embeddings import NT_DICT
from sincfold.utils import write_ct, validate_file, ct2dot
from sincfold.parser import parser
from sincfold.utils import dot2png, ct2svg

from sincfold.model import sincfold
from sincfold.ablation.ablation_C1D_C2D import sincfold_C1D_C2D
from sincfold.ablation.ablation_no_ResNet2d import sincfold_no_ResNet2d
from sincfold.ablation.ablation_1ResNet2d import sincfold_1ResNet2d
from sincfold.ablation.ablation_no_ResNet1d import sincfold_no_ResNet1d
from sincfold.ablation.ablation_no_ResNet1d_FF import sincfold_no_ResNet1d_FF


def test_net(ablation, test_file, model_weights=None, output_file=None, config={}, nworkers=2, verbose=True):
    model = { 'sincfold' : sincfold, 
        '1ResNet2d' : sincfold_1ResNet2d, 
        'no_ResNet1d_FF' : sincfold_no_ResNet1d_FF,
        'no_ResNet2d' : sincfold_no_ResNet2d,
        'C1D_C2D' : sincfold_C1D_C2D,
        'no_ResNet1d' :sincfold_no_ResNet1d}

    test_file = test_file
    test_file = validate_file(test_file)
    if verbose not in config:
        config["verbose"] = verbose

    test_loader = DataLoader(
        SeqDataset(test_file, **config),
        batch_size=config["batch_size"] if "batch_size" in config else 4,
        shuffle=False,
        num_workers=nworkers,
        collate_fn=pad_batch,
    )

    if model_weights is not None:
        net = model[ablation](weights=model_weights, **config)
    else:
        net = model[ablation](pretrained=True, **config)
    
    if verbose:
        print(f"Start test of {test_file}")        
    test_metrics = net.test(test_loader)
    summary = ",".join([k for k in sorted(test_metrics.keys())]) + "\n" + ",".join([f"{test_metrics[k]:.3f}" for k in sorted(test_metrics.keys())])+ "\n" 
    if output_file is not None:
        with open(output_file, "w") as f:
            f.write(summary)
    if verbose:
        print(summary)