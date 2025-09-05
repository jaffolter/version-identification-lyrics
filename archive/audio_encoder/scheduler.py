import numpy as np
import math 

# Code taken from 
# https://github.com/LAION-AI/CLAP/blob/main/src/laion_clap/training/scheduler.py

def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster


def cosine_with_warmup(step, cfg, len_train_loader):
    warmup = cfg.optimizer.warmup_steps
    total = cfg.train.epochs * len_train_loader  # total steps

    if step < warmup:
        return step / warmup
    else:
        progress = (step - warmup) / (total - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))
    