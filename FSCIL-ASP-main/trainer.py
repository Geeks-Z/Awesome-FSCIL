import sys
import logging
import copy
import time

import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import random
import numpy as np
import pickle


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    # 获取 GPU 型号信息
    gpu_names = []
    for dev in device:
        if dev.isdigit():  # 检查是否是 GPU 设备编号
            idx = int(dev)
            if idx < torch.cuda.device_count():
                gpu_names.append(torch.cuda.get_device_name(idx))
    args["gpu_models"] = gpu_names  # 将 GPU 型号列表添加到 args

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):

    init_cls = args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}_{}".format(
        args["model_name"], args["dataset"], init_cls, args["increment"], args["kshot"]
    )
    saved_path = "saved_model/{}/{}/{}_{}".format(
        args["model_name"], args["dataset"], init_cls, args["increment"]
    )

    if not os.path.exists(logs_name):
        os.makedirs(logs_name)
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)

    logfilename = "logs/{}/{}/{}/{}_{}/{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args["kshot"],
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

    args["base_model_path"] = "saved_model/{}/{}/{}_{}/{}_{}_{}_{}.pth".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args["model_prefix"],
        args["tuned_epoch"],
        args["seed"],
        args["backbone_type"],
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
    
    # 初始化训练统计字典
    train_stats = {
        'task': [],
        'params': [],
        'gpu': [],
        'time': [],
        'epochs': []
    }
    
    for task in range(data_manager.nb_tasks):
        # 统计可训练参数(不包括分类器fc和future_head)
        total_trainable_params = 0
        for name, param in model._network.named_parameters():
            if param.requires_grad and 'fc.' not in name and 'future_head.' not in name:
                total_trainable_params += param.numel()
        
        # 记录参数量
        train_stats['params'].append(total_trainable_params)
        
        # 重置GPU显存统计（使用模型所在的设备）
        if torch.cuda.is_available() and hasattr(model._network, 'device'):
            # 获取模型所在的设备
            device = model._network.device if hasattr(model._network.device, 'index') else torch.device('cuda:0')
            device_id = device.index if hasattr(device, 'index') else 0
            torch.cuda.reset_peak_memory_stats(device=device_id)
            torch.cuda.empty_cache()
        elif torch.cuda.is_available():
            # 如果无法获取模型设备，使用第一个可用GPU
            device_id = int(str(args['device'][0]).split(':')[-1]) if len(args['device']) > 0 else 0
            torch.cuda.reset_peak_memory_stats(device=device_id)
            torch.cuda.empty_cache()

        model.incremental_train(data_manager)
        
        # 使用模型记录的纯训练时间
        task_time = model.train_time
        total_train_time += task_time
        
        # 获取GPU峰值显存（使用模型所在的设备）
        if torch.cuda.is_available() and hasattr(model._network, 'device'):
            device = model._network.device if hasattr(model._network.device, 'index') else torch.device('cuda:0')
            device_id = device.index if hasattr(device, 'index') else 0
            peak_memory = torch.cuda.max_memory_allocated(device=device_id) / (1024 ** 2)  # 转换为MiB
        elif torch.cuda.is_available():
            device_id = int(str(args['device'][0]).split(':')[-1]) if len(args['device']) > 0 else 0
            peak_memory = torch.cuda.max_memory_allocated(device=device_id) / (1024 ** 2)
        else:
            peak_memory = 0
        
        # 记录统计信息
        train_stats['task'].append(task)
        train_stats['gpu'].append(peak_memory)
        train_stats['time'].append(task_time)
        train_stats['epochs'].append(model.last_epochs)
        
        # 打印当前任务的可训练参数量
        logging.info(f"Task {task} Trainable params: {total_trainable_params / 1e6:.2f}M, GPU: {peak_memory:.2f}MiB")

        test_start_time = time.time()
        cnn_accy, prompt_time = model.eval_task()
        total_prompt_time += prompt_time

        total_test_time += time.time() - test_start_time
        model.after_task()

        cnn_keys = [key for key in cnn_accy["grouped"].keys() if "-" in key]
        cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
        cnn_matrix.append(cnn_values)

        logging.info("CNN: {}".format(cnn_accy["grouped"]))

        cnn_curve["top1"].append(cnn_accy["top1"])

        logging.info("Top1 curve: {}".format(cnn_curve["top1"]))

        logging.info(
            "Average Accuracy (CNN): {:.2f}".format(
                sum(cnn_curve["top1"]) / len(cnn_curve["top1"])
            )
        )

        # Hacc, old_acc, new_acc = Harmonic_Accuracy(
        #     cnn_accy["grouped"], args["init_cls"]
        # )
        # logging.info(
        #     "Average Accuracy (Top1): {}   (Harmonic Accuracy): {} (Old Acc): {} (New Acc): {} \n".format(
        #         sum(cnn_curve["top1"]) / len(cnn_curve["top1"]), Hacc, old_acc, new_acc
        #     )
        # )

    print(f"\n{'=' * 80}")
    print(
        "Finished {}_init{}_inc{}: {}  ".format(
            args["dataset"],
            args["init_cls"],
            args["increment"],
            args["backbone_type"],
        )
    )
    print("Base Accuracy: {}".format(round(cnn_curve["top1"][0], 2)))
    print("Last Accuracy: {}".format(round(cnn_curve["top1"][-1], 2)))
    print("Average Accuracy (Top1): {}".format(round(sum(cnn_curve["top1"]) / len(cnn_curve["top1"]), 2)))
    print("PD: {:.2f}".format(cnn_curve["top1"][0] - cnn_curve["top1"][-1]))
    print("Backward Transfer (BWT):", backward_transfer(np.array(cnn_matrix)))
    print("Forward Transfer (FWT):", forward_transfer(args["dataset"], np.array(cnn_matrix)))
    # print("Total Train Time:", round(total_train_time, 2), "s")
    # print("Total Test Time:", round(total_test_time, 2), "s")
    if len(cnn_matrix) > 0:
        np_acctable = np.zeros([task + 1, task + 1])
        for idxx, line in enumerate(cnn_matrix):
            idxy = len(line)
            np_acctable[idxx, :idxy] = np.array(line)
        print("Accuracy Matrix (CNN):")
        print(np_acctable)

    print(f"{'=' * 80}\n")
    
    # 打印训练统计信息（使用独立的分隔线）
    print("\n" + "=" * 80)
    print_training_statistics(train_stats, args)
    print("=" * 80 + "\n")


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

def backward_transfer(matrix):
    """
    Calculate the backward transfer from the accuracy matrix.
    """

    # --- Backward Transfer (BWT) ---
    # 根据公式: BWT = (1/(T-1)) * Σ(i=1 to T-1) (R_T,i - R_i,i)
    # R_T,i 是训练完最后一个任务 T 后，在任务 i 上的准确率 -> 对应矩阵的最后一行
    # R_i,i 是刚训练完任务 i 后的准确率 -> 对应矩阵的对角线

    # 使用 0-based 索引:
    # R_T,i (i=1..T-1) -> matrix[-1, 0:T-1]
    # R_i,i (i=1..T-1) -> np.diag(matrix)[0:T-1]

    last_row_accs = matrix[-1, :-1]  # 任务 1..T-1 在训练完任务 T 后的准确率
    diagonal_accs = np.diag(matrix)[:-1]  # 任务 1..T-1 在刚训练完自己时的准确率

    bwt_diffs = last_row_accs - diagonal_accs
    backward_transfer = np.mean(bwt_diffs)

    # print("--- Backward Transfer (BWT) ---")
    # print(f"各任务在最终的准确率 (R_T,i): {np.round(last_row_accs, 2)}")
    # print(f"各任务在当时的准确率 (R_i,i): {np.round(diagonal_accs, 2)}")
    # print(f"BWT 差值 (R_T,i - R_i,i): {np.round(bwt_diffs, 2)}")
    # print(f"Backward Transfer 平均分: {backward_transfer:.2f}")

    return np.round(backward_transfer, 2)

def forward_transfer(dataset, matrix):
    """
    Calculate the forward transfer from the accuracy matrix.
    """
    if dataset == "cub":
        rand_init_acc = np.array([86.51, 52.53, 65.04, 67.86, 74.62, 57.89, 71.82, 87.97, 68.6, 83.61, 85.37])
    elif dataset == "cifar224":
        rand_init_acc = np.array([77.5, 44.4, 51.8, 30.4, 56.8, 68.4, 44.0, 36.4, 41.4])
    elif dataset == "mini_imagenet":
        rand_init_acc = np.array([94.97, 90.6, 69.2, 81.2, 82.8, 73.4, 70.2, 90.6, 89.8])
    else:
        rand_init_acc = np.array([62.95, 28.08, 51.15, 38.64, 56.76, 50.93, 41.38, 61.25, 49.1, 43.37, 44.44])  # INR

    fwt_accs = np.diag(matrix, k=1)       # R_0,1, R_1,2, ...
    rand_init_acc_for_fwt = rand_init_acc[1:]  # R_0,1, R_0,2, ...

    # 维度不匹配时直接返回提示，不再计算，避免报错
    if fwt_accs.shape[0] != rand_init_acc_for_fwt.shape[0]:
        msg = f"FWT 计算失败：矩阵长度不匹配。"
        # print(msg)
        return msg

    fwt_diffs = fwt_accs - rand_init_acc_for_fwt
    forward_transfer_value = np.mean(fwt_diffs)

    return np.round(forward_transfer_value, 2)


def print_training_statistics(train_stats, args):
    """打印训练统计信息"""
    print("\nTraining Statistics")
    print("-" * 80)
    print(f"{'Task':<8} {'Params ↓':<14} {'GPU ↓':<14} {'Time (s/epoch)':<12}")
    print("-" * 80)
    
    for i in range(len(train_stats['task'])):
        task = train_stats['task'][i]
        params = train_stats['params'][i] / 1e6  # 转换为M
        gpu = train_stats['gpu'][i]
        time_total = train_stats['time'][i]
        epochs = train_stats['epochs'][i]
        
        # 计算每epoch时间
        time_per_epoch = time_total / epochs if epochs > 0 else 0
        
        print(f"Task {task:<3} {params:.2f}M{'':<6} {gpu:.0f}MiB{'':<8} {time_per_epoch:.2f}s")
    
    # 计算平均值和总计
    avg_params = sum(train_stats['params']) / len(train_stats['params']) / 1e6
    total_params = sum(train_stats['params']) / 1e6  # 累计总参数量
    avg_gpu = sum(train_stats['gpu']) / len(train_stats['gpu'])
    max_gpu = max(train_stats['gpu'])
    total_time = sum(train_stats['time'])
    
    # 计算平均每epoch时间
    total_epochs = sum(train_stats['epochs'])
    avg_time_per_epoch = total_time / total_epochs if total_epochs > 0 else 0
    
    print("-" * 80)
    print(f"{'Average':<8} {avg_params:.2f}M{'':<6} {avg_gpu:.0f}MiB{'':<8} {avg_time_per_epoch:.2f}s")
    print(f"{'Max GPU':<8} {'':<14} {max_gpu:.0f}MiB")
    print(f"{'Total':<8} {total_params:.2f}M{'':<6} {'':<14} {total_time:.2f}s")

