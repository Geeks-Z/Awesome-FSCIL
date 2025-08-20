import sys
import logging
import copy
import time

import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import numpy as np


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):

    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"],args["dataset"], init_cls, args['increment'])
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args["prefix"],
        args["seed"],
        args["backbone_type"],
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random(args["seed"])
    _set_device(args)
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

    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    cnn_matrix, nme_matrix = [], []

    total_train_time = 0.0
    total_test_time = 0.0

    for task in range(data_manager.nb_tasks):
        # task = 9
        print('task',task)
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        start_time = time.time()
        model.incremental_train(data_manager)
        train_end_time = time.time()
        total_train_time += (train_end_time - start_time)
        cnn_accy, nme_accy = model.eval_task()
        total_test_time += (time.time() - train_end_time)
        model.after_task()

        cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
        cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
        cnn_matrix.append(cnn_values)

        logging.info("CNN: {}".format(cnn_accy["grouped"]))

        cnn_curve["top1"].append(cnn_accy["top1"])

        logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))

        logging.info(
            "Average Accuracy (CNN): {} \n".format(round(sum(cnn_curve["top1"]) / len(cnn_curve["top1"]), 2)))

    print(f"\n{'=' * 100}")
    print(
        "Finished {}_init{}_inc{}: {}  ".format(
            args["dataset"],
            args["init_cls"],
            args["increment"],
            args["backbone_type"],
        )
    )
    print('Average Accuracy (CNN):', round(sum(cnn_curve["top1"]) / len(cnn_curve["top1"]), 2))
    print("Total Train Time:", round(total_train_time, 2), "s")
    print("Total Test Time:", round(total_test_time, 2), "s")
    if len(cnn_matrix) > 0:
        np_acctable = np.zeros([task + 1, task + 1])
        for idxx, line in enumerate(cnn_matrix):
            idxy = len(line)
            np_acctable[idxx, :idxy] = np.array(line)
        print("Accuracy Matrix (CNN):")
        print(np_acctable)

    print(f"{'=' * 100}\n")

def _set_device(args):
    device_type = args["device"]
    gpus = []
    print('devices_type', device_type)
    for device in device_type:
        if device == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))