import sys
import logging
import copy
import time
import os
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import random
import numpy as np


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):
    init_cls = args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}_{}/{}".format(
        "sec_tr",
        args["dataset"],
        args["tuned_epoch"],
        args["init_lr"],
        args["kshot"],
        args["beta"],
    )
    saved_path = "saved_model/{}/{}/{}_{}/{}_{}".format(
        "sec_tr",
        args["dataset"],
        args["tuned_epoch"],
        args["init_lr"],
        args["prompt_token_num"],
        args["prompt_pool_num"],
    )

    if not os.path.exists(logs_name):
        os.makedirs(logs_name)
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)

    logfilename = "logs/{}/{}/{}/{}_{}/{}/{}_{}".format(
        "sec_tr",
        args["dataset"],
        args["tuned_epoch"],
        args["init_lr"],
        args["kshot"],
        args["beta"],
        args["prompt_token_num"],
        args["prompt_pool_num"],
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    args["base_model_path"] = "saved_model/{}/{}/{}_{}/{}_{}/{}_{}_{}_{}.pth".format(
        "sec_tr",
        args["dataset"],
        args["tuned_epoch"],
        args["init_lr"],
        args["prompt_token_num"],
        args["prompt_pool_num"],
        args["model_prefix"],
        args["tuned_epoch"],
        args["seed"],
        args["batch_size"],
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

    args["nb_classes"] = data_manager.nb_classes  # update args
    args["nb_tasks"] = data_manager.nb_tasks
    model = factory.get_model(args["model_name"], args)

    cnn_curve = {"top1": [], "top5": []}
    cnn_matrix = []
    total_train_time = 0.0
    total_test_time = 0.0
    total_prompt_time = 0.0
    for task in range(data_manager.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )

        start_time = time.time()

        model.incremental_train(data_manager)

        train_end_time = time.time()
        total_train_time += train_end_time - start_time

        cnn_accy = model.eval_task()
        total_prompt_time += cnn_accy.get("prompt_time", 0)

        total_test_time += time.time() - train_end_time
        model.after_task()

        cnn_keys = [key for key in cnn_accy["grouped"].keys() if "-" in key]
        cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
        cnn_matrix.append(cnn_values)

        cnn_curve["top1"].append(cnn_accy["top1"])

        logging.info("Top1 curve: {}".format(cnn_curve["top1"]))

        # Hacc, old_acc, new_acc = Harmonic_Accuracy(
        #     cnn_accy["grouped"], args["init_cls"]
        # )
        # logging.info(
        #     "Average Accuracy (Top1): {}   (Harmonic Accuracy): {} (Old Acc): {} (New Acc): {} \n".format(
        #         sum(cnn_curve["top1"]) / len(cnn_curve["top1"]), Hacc, old_acc, new_acc
        #     )
        # )

    print(f"\n{'=' * 100}")
    print(
        "Finished {}_init{}_inc{}: {}  ".format(
            args["dataset"],
            args["init_cls"],
            args["increment"],
            args["backbone_type"],
        )
    )
    print("Average Accuracy (Top1): {}".format(round(sum(cnn_curve["top1"]) / len(cnn_curve["top1"]),2)))
    print("Total Train Time:", round(total_train_time, 2), "s")
    print("Total Test Time:", round(total_test_time, 2), "s")
    print("Total Prompt Time:", round(total_prompt_time, 2), "s")

    if len(cnn_matrix) > 0:
        np_acctable = np.zeros([task + 1, task + 1])
        for idxx, line in enumerate(cnn_matrix):
            idxy = len(line)
            np_acctable[idxx, :idxy] = np.array(line)
        np_acctable = np_acctable.T
        forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
        print('Accuracy Matrix (CNN):')
        print(np_acctable)
        logging.info('Forgetting (CNN): {}'.format(forgetting))

    if len(cnn_matrix) > 0:
        np_acctable = np.zeros([task + 1, task + 1])
        for idxx, line in enumerate(cnn_matrix):
            idxy = len(line)
            np_acctable[:idxy, idxx] = np.array(line)
        # np_acctable = np_acctable.T  <- 这行不再需要
        forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
        print('Accuracy Matrix (CNN):')
        print(np_acctable)
        logging.info('Forgetting (CNN): {}'.format(forgetting))

    print(f"{'=' * 100}\n")


def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device_type == -1:
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
    random.seed(seed)
    np.random.seed(seed)


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))


def Harmonic_Accuracy(grouped_acc, init_cls):
    old_acc, new_acc = [], []
    for key in grouped_acc.keys():
        if "-" in key:
            if int(key.split("-")[1]) < init_cls:
                old_acc.append(grouped_acc[key])
            elif int(key.split("-")[1]) > init_cls:
                new_acc.append(grouped_acc[key])
    old_acc = sum(old_acc) / len(old_acc)

    if len(new_acc) > 0:
        new_acc = sum(new_acc) / len(new_acc)
        Hacc = 2 * old_acc * new_acc / (old_acc + new_acc)
    else:
        Hacc = None
    return Hacc, old_acc, new_acc
