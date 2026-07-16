import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import random
import numpy as np
import pickle
import inspect
import shutil

def get_class_file_path(instance):
    module = inspect.getmodule(instance.__class__)
    if module:
        file_path = os.path.abspath(module.__file__)
        return file_path
    else:
        return None

def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = torch.device('cuda:'+args["device"][0])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):

    init_cls = args["init_cls"]
    saved_path = "logs/{}/{}/{}_{}_k{}_s{}_{}".format(args["model_name"],args["dataset"], init_cls, args['increment'], args["kshot"], args["seed"], args["tag"])
    log_name = 'log'
    if args['special_flag']: # 替换save path
        # ********** previous version
        # args['base_model_path'] = os.path.join(saved_path, 'session_0.pth')
        # # assert os.path.exists(args['base_model_path']), f"文件不存在: {args['base_model_path']}"
        # saved_path = os.path.join(os.path.dirname(saved_path), args['special_tag'] + "_" + os.path.basename(saved_path))

        # ********** 0619 version for calibrate analysis
        args['base_model_path_init'] = os.path.join(saved_path, 'session_0.pth')
        args['base_model_path'] = os.path.join(saved_path, 'session_10.pth' if args['init_cls'] == 100 else 'session_8.pth')
        # assert os.path.exists(args['base_model_path']), f"文件不存在: {args['base_model_path']}"
        log_name = 'special_log'
    args["saved_path"] = saved_path
    logfilename = os.path.join(saved_path,log_name)

    if not os.path.exists(saved_path):
        os.makedirs(saved_path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random(args["seed"])
    # _set_device(args)
    print_args(args)

    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args,
    )
    
    args["nb_classes"] = data_manager.nb_classes # update args
    args["nb_tasks"] = data_manager.nb_tasks
    model = factory.get_model(args["model_name"], args)

    # save python method for backup
    model_path = get_class_file_path(model)
    network_path = get_class_file_path(model._network)
    block_path = get_class_file_path(model._network.backbone.blocks[0])

    if args['special_flag']:
        shutil.copy(model_path, os.path.join(saved_path,'special_'+os.path.basename(model_path)))
        shutil.copy(network_path, os.path.join(saved_path,'special_'+os.path.basename(network_path)))
        shutil.copy(block_path, os.path.join(saved_path,'special_'+os.path.basename(block_path)))
    else:
        shutil.copy(model_path, saved_path)
        shutil.copy(network_path, saved_path)
        shutil.copy(block_path, saved_path)


    top1_curve = {"top1": [], "top5": []}
    for task in range(data_manager.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        
        model.incremental_train(data_manager)
        if args['special_flag'] and task < data_manager.nb_tasks - 1:
            model.after_task()
            continue

        top1_accy = model.eval_task()
        model.after_task()

        top1_curve["top1"].append(top1_accy["top1"])

        logging.info("Top1 curve: {}".format(top1_curve["top1"]))
        
        Hacc, old_acc, new_acc = Harmonic_Accuracy(top1_accy["grouped"], args["init_cls"])
        logging.info("Average Accuracy (Top1): {}   (Harmonic Accuracy): {} (Old Acc): {} (New Acc): {} \n".format(sum(top1_curve["top1"])/len(top1_curve["top1"]),
                                                                            Hacc, old_acc, new_acc))

        if args['special_flag']:
            model.calibrate_analysis()

    logging.info("\n")

    
# def _set_device(args):
#     device_type = args["device"]
#     gpus = []
#
#     for device in device_type:
#         if device_type == -1:
#             device = torch.device("cpu")
#         else:
#             device = torch.device("cuda:{}".format(device))
#
#         gpus.append(device)
#
#     args["device"] = gpus


def _set_random(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))

def Harmonic_Accuracy(grouped_acc, init_cls):
    old_acc, new_acc = [], []
    for key in grouped_acc.keys():
        if '-' in key:  
            if int(key.split('-')[1]) < init_cls:
                old_acc.append(grouped_acc[key])
            elif int(key.split('-')[1]) > init_cls:
                new_acc.append(grouped_acc[key])
    old_acc = sum(old_acc) / len(old_acc)

    if len(new_acc) > 0:
        new_acc = sum(new_acc) / len(new_acc)
        Hacc = 2 * old_acc * new_acc / (old_acc + new_acc)
    else:
        Hacc = None
    return Hacc, old_acc, new_acc
