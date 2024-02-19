import os
import math
import time
from Utils.option import *

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def create_log(if_train=True, iter=True):
    if if_train and not iter:
        create_dir('./Log/TrainLog/')
        Logger = open(f'./Log/TrainLog/{opt.trainset_name}_train_epoch.log', 'a+')
    elif if_train and iter:
        create_dir('./Log/TrainLog/')
        Logger = open(f'./Log/TrainLog/{opt.trainset_name}_train_iter.log', 'a+')
    else:
        create_dir('./Log/TestLog/')
        Logger = open(f'./Log/TestLog/{opt.trainset_name}_train.log', 'a+')
    Logger.write(time.strftime('\n%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    Logger.write('\n')
    return Logger

def close_log(Logger):
    Logger.write(time.strftime('\n%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    Logger.close()

def lr_schedule_cosdecay(t, T, init_lr=0.0001):
    lr = 0.5 * (1 + math.cos(t * math.pi / T)) * init_lr
    return lr

def adjust_learning_rate(epoch, optim):
    if opt.lr_sche:
        lr = lr_schedule_cosdecay(epoch, opt.num_epochs, init_lr=opt.lr)
    else:
        lr = opt.lr
    for param_group in optim.param_groups:
        param_group["lr"] = lr
    return lr




