import sys
import logging
import copy
import time
import os
import random

import torch
import numpy as np

from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters


def _resolve_cuda_device_id(args):
    if not torch.cuda.is_available():
        return None

    for device in args["device"]:
        if isinstance(device, torch.device) and device.type == "cuda":
            return 0 if device.index is None else device.index

    return 0


def _create_stage_metrics():
    return {
        "stage_id": [],
        "train_params_trainable_m": [],
        "train_gpu_peak_mib": [],
        "train_epochs": [],
        "train_time_stage_s": [],
        "train_time_epoch_s": [],
        "infer_params_total_m": [],
        "infer_gpu_peak_mib": [],
        "infer_num_samples": [],
        "infer_time_stage_s": [],
        "infer_latency_ms_per_img": [],
        "infer_prompt_time_stage_s": [],
    }


def _stage_name(stage_id):
    return "S{}".format(stage_id)


def _safe_mean(values):
    return sum(values) / len(values) if values else 0.0


def _count_loader_samples(loader):
    if loader is None:
        return 0

    dataset = getattr(loader, "dataset", None)
    if dataset is not None:
        try:
            return len(dataset)
        except TypeError:
            pass

    try:
        return len(loader)
    except TypeError:
        return 0


def _resolve_infer_sample_count(model):
    future_eval_loader = getattr(model, "future_test_loader", getattr(model, "future_loader", None))
    return _count_loader_samples(getattr(model, "test_loader", None)) + _count_loader_samples(future_eval_loader)


def _resolve_stage_epochs(args, stage_id, model=None):
    model_epochs = getattr(model, "last_epochs", None)
    if isinstance(model_epochs, (int, float)) and model_epochs > 0:
        return int(model_epochs)

    if stage_id == 0 and "init_epoch" in args:
        return int(args["init_epoch"])

    if stage_id > 0 and "epochs" in args:
        return int(args["epochs"])

    if stage_id > 0 and "fs_epoch" in args:
        return int(args["fs_epoch"])

    if "tuned_epoch" in args:
        return int(args["tuned_epoch"])

    return 0


def _parameter_name_matches_prefix(name, prefix):
    normalized_name = name[7:] if name.startswith("module.") else name
    return normalized_name == prefix or normalized_name.startswith(prefix + ".")


def _resolve_metric_excluded_param_prefixes(model):
    prefixes = ["future_head", "original_backbone"]
    prefixes.extend(getattr(model, "metric_excluded_param_prefixes", []))
    return tuple(prefixes)


def _resolve_stage_trainable_param_names(model):
    if hasattr(model, "get_stage_trainable_param_names"):
        param_names = model.get_stage_trainable_param_names()
        return list(param_names) if param_names is not None else []

    param_names = getattr(model, "stage_trainable_param_names", None)
    if param_names is not None:
        return list(param_names)

    excluded_prefixes = _resolve_metric_excluded_param_prefixes(model)
    return [
        name
        for name, param in model._network.named_parameters()
        if param.requires_grad
        and not any(_parameter_name_matches_prefix(name, prefix) for prefix in excluded_prefixes)
    ]


def _sum_named_parameters(module, param_names):
    named_parameters = dict(module.named_parameters())
    return sum(named_parameters[name].numel() for name in param_names if name in named_parameters)


def _resolve_stage_trainable_stats(model):
    if hasattr(model, "get_stage_trainable_params_m"):
        return float(model.get_stage_trainable_params_m()), []

    param_names = _resolve_stage_trainable_param_names(model)
    return _sum_named_parameters(model._network, param_names) / 1e6, param_names


def _collect_param_names_by_prefixes(module, prefixes):
    selected = []
    for name, _ in module.named_parameters():
        if any(_parameter_name_matches_prefix(name, prefix) for prefix in prefixes):
            selected.append(name)
    return selected


def _resolve_inference_added_params_m(model, args):
    prefixes = ("backbone.TSP", "backbone.RSP", "fc")
    param_names = _collect_param_names_by_prefixes(model._network, prefixes)
    return _sum_named_parameters(model._network, param_names) / 1e6


def _parse_eval_result(eval_result):
    cnn_accy = eval_result
    prompt_time_stage_s = 0.0

    if isinstance(eval_result, tuple):
        cnn_accy = eval_result[0]
        if len(eval_result) > 1 and isinstance(eval_result[1], (int, float)):
            prompt_time_stage_s = float(eval_result[1])

    if isinstance(cnn_accy, dict):
        prompt_time_stage_s = float(
            cnn_accy.get("prompt_time", prompt_time_stage_s) or prompt_time_stage_s
        )

    return cnn_accy, prompt_time_stage_s


def _append_stage_metrics(
    stage_metrics,
    stage_id,
    model,
    args,
    train_params_trainable_m,
    train_time_stage_s,
    train_gpu_peak_mib,
    infer_params_total_m,
    infer_time_stage_s,
    infer_gpu_peak_mib,
    infer_num_samples,
    prompt_time_stage_s=0.0,
):
    train_epochs = _resolve_stage_epochs(args, stage_id, model)
    train_time_epoch_s = train_time_stage_s / train_epochs if train_epochs > 0 else 0.0
    infer_latency_ms_per_img = (
        infer_time_stage_s * 1000.0 / infer_num_samples if infer_num_samples > 0 else 0.0
    )

    stage_metrics["stage_id"].append(stage_id)
    stage_metrics["train_params_trainable_m"].append(train_params_trainable_m)
    stage_metrics["train_gpu_peak_mib"].append(train_gpu_peak_mib)
    stage_metrics["train_epochs"].append(train_epochs)
    stage_metrics["train_time_stage_s"].append(train_time_stage_s)
    stage_metrics["train_time_epoch_s"].append(train_time_epoch_s)
    stage_metrics["infer_params_total_m"].append(infer_params_total_m)
    stage_metrics["infer_gpu_peak_mib"].append(infer_gpu_peak_mib)
    stage_metrics["infer_num_samples"].append(infer_num_samples)
    stage_metrics["infer_time_stage_s"].append(infer_time_stage_s)
    stage_metrics["infer_latency_ms_per_img"].append(infer_latency_ms_per_img)
    stage_metrics["infer_prompt_time_stage_s"].append(prompt_time_stage_s)


def _print_training_metrics(stage_metrics, final_trainable_total_m):
    print("Training Task Metrics")
    print("-" * 84)
    print(
        f"{'Task':<8}{'Trainable(M)':<16}{'GPU(MiB)':<12}"
        f"{'Time/Task(s)':<15}{'Epochs':<8}{'Time/Epoch(s)':<15}"
    )
    print("-" * 84)

    for idx, stage_id in enumerate(stage_metrics["stage_id"]):
        print(
            f"{_stage_name(stage_id):<8}"
            f"{stage_metrics['train_params_trainable_m'][idx]:<16.2f}"
            f"{stage_metrics['train_gpu_peak_mib'][idx]:<12.2f}"
            f"{stage_metrics['train_time_stage_s'][idx]:<15.2f}"
            f"{stage_metrics['train_epochs'][idx]:<8}"
            f"{stage_metrics['train_time_epoch_s'][idx]:<15.2f}"
        )

    print("-" * 84)
    base_idx = 0
    inc_slice = slice(1, None)
    print(
        f"{'Base S0':<8}"
        f"{stage_metrics['train_params_trainable_m'][base_idx]:<16.2f}"
        f"{stage_metrics['train_gpu_peak_mib'][base_idx]:<12.2f}"
        f"{stage_metrics['train_time_stage_s'][base_idx]:<15.2f}"
        f"{stage_metrics['train_epochs'][base_idx]:<8}"
        f"{stage_metrics['train_time_epoch_s'][base_idx]:<15.2f}"
    )

    if len(stage_metrics["stage_id"]) > 1:
        print(
            f"{'Inc Avg':<8}"
            f"{_safe_mean(stage_metrics['train_params_trainable_m'][inc_slice]):<16.2f}"
            f"{_safe_mean(stage_metrics['train_gpu_peak_mib'][inc_slice]):<12.2f}"
            f"{_safe_mean(stage_metrics['train_time_stage_s'][inc_slice]):<15.2f}"
            f"{_safe_mean(stage_metrics['train_epochs'][inc_slice]):<8.2f}"
            f"{_safe_mean(stage_metrics['train_time_epoch_s'][inc_slice]):<15.2f}"
        )

    print(
        f"{'All Avg':<8}"
        f"{_safe_mean(stage_metrics['train_params_trainable_m']):<16.2f}"
        f"{_safe_mean(stage_metrics['train_gpu_peak_mib']):<12.2f}"
        f"{_safe_mean(stage_metrics['train_time_stage_s']):<15.2f}"
        f"{_safe_mean(stage_metrics['train_epochs']):<8.2f}"
        f"{_safe_mean(stage_metrics['train_time_epoch_s']):<15.2f}"
    )
    print(
        f"{'Max GPU':<8}{'':<16}{max(stage_metrics['train_gpu_peak_mib']):<12.2f}"
        f"{'':<15}{'':<8}{'':<15}"
    )
    print(
        f"{'Total':<8}{'':<28}{'':<12}"
        f"{sum(stage_metrics['train_time_stage_s']):<15.2f}"
    )
    print(f"{'Final':<8}{final_trainable_total_m:<16.2f}{'':<12}{'':<15}{'':<8}{'':<15}")


def _print_inference_metrics(stage_metrics):
    print("Inference Task Metrics")
    print("-" * 90)
    print(
        f"{'Task':<8}{'Added(M)':<12}{'GPU(MiB)':<12}{'Time/Task(s)':<15}"
        f"{'Latency(ms/img)':<18}{'Samples':<10}{'Prompt(s)':<12}"
    )
    print("-" * 90)

    for idx, stage_id in enumerate(stage_metrics["stage_id"]):
        print(
            f"{_stage_name(stage_id):<8}"
            f"{stage_metrics['infer_params_total_m'][idx]:<12.2f}"
            f"{stage_metrics['infer_gpu_peak_mib'][idx]:<12.2f}"
            f"{stage_metrics['infer_time_stage_s'][idx]:<15.4f}"
            f"{stage_metrics['infer_latency_ms_per_img'][idx]:<18.4f}"
            f"{stage_metrics['infer_num_samples'][idx]:<10}"
            f"{stage_metrics['infer_prompt_time_stage_s'][idx]:<12.4f}"
        )

    print("-" * 90)
    base_idx = 0
    inc_slice = slice(1, None)
    print(
        f"{'Base S0':<8}"
        f"{stage_metrics['infer_params_total_m'][base_idx]:<12.2f}"
        f"{stage_metrics['infer_gpu_peak_mib'][base_idx]:<12.2f}"
        f"{stage_metrics['infer_time_stage_s'][base_idx]:<15.4f}"
        f"{stage_metrics['infer_latency_ms_per_img'][base_idx]:<18.4f}"
        f"{stage_metrics['infer_num_samples'][base_idx]:<10}"
        f"{stage_metrics['infer_prompt_time_stage_s'][base_idx]:<12.4f}"
    )

    if len(stage_metrics["stage_id"]) > 1:
        print(
            f"{'Inc Avg':<8}"
            f"{_safe_mean(stage_metrics['infer_params_total_m'][inc_slice]):<12.2f}"
            f"{_safe_mean(stage_metrics['infer_gpu_peak_mib'][inc_slice]):<12.2f}"
            f"{_safe_mean(stage_metrics['infer_time_stage_s'][inc_slice]):<15.4f}"
            f"{_safe_mean(stage_metrics['infer_latency_ms_per_img'][inc_slice]):<18.4f}"
            f"{_safe_mean(stage_metrics['infer_num_samples'][inc_slice]):<10.2f}"
            f"{_safe_mean(stage_metrics['infer_prompt_time_stage_s'][inc_slice]):<12.4f}"
        )

    print(
        f"{'All Avg':<8}"
        f"{_safe_mean(stage_metrics['infer_params_total_m']):<12.2f}"
        f"{_safe_mean(stage_metrics['infer_gpu_peak_mib']):<12.2f}"
        f"{_safe_mean(stage_metrics['infer_time_stage_s']):<15.4f}"
        f"{_safe_mean(stage_metrics['infer_latency_ms_per_img']):<18.4f}"
        f"{_safe_mean(stage_metrics['infer_num_samples']):<10.2f}"
        f"{_safe_mean(stage_metrics['infer_prompt_time_stage_s']):<12.4f}"
    )
    print(
        f"{'Max GPU':<8}{'':<12}{max(stage_metrics['infer_gpu_peak_mib']):<12.2f}"
        f"{'':<15}{'':<18}{'':<10}{'':<12}"
    )
    print(
        f"{'Total':<8}{'':<24}"
        f"{sum(stage_metrics['infer_time_stage_s']):<15.4f}"
        f"{'':<18}{'':<10}"
        f"{sum(stage_metrics['infer_prompt_time_stage_s']):<12.4f}"
    )


def print_stage_metric_tables(stage_metrics, final_trainable_total_m):
    if not stage_metrics["stage_id"]:
        return

    print("\n" + "=" * 90)
    _print_training_metrics(stage_metrics, final_trainable_total_m)
    print("-" * 90)
    _print_inference_metrics(stage_metrics)
    print("=" * 90 + "\n")


def _resolve_final_trainable_total_m(model, cumulative_trainable_param_names):
    if hasattr(model, "get_final_trainable_params_m"):
        return float(model.get_final_trainable_params_m())
    return _sum_named_parameters(model._network, cumulative_trainable_param_names) / 1e6


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

    _set_device(args)
    _set_random(args["seed"], args)
    print_args(args)

    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args,
    )

    args["nb_classes"] = data_manager.nb_classes
    args["nb_tasks"] = data_manager.nb_tasks
    model = factory.get_model(args["model_name"], args)

    cnn_curve = {"top1": [], "top5": []}
    cnn_matrix = []
    total_train_time = 0.0
    total_test_time = 0.0
    total_prompt_time = 0.0
    stage_metrics = _create_stage_metrics()
    cumulative_trainable_param_names = set()

    for task in range(data_manager.nb_tasks):
        device_id = _resolve_cuda_device_id(args)
        if device_id is not None:
            torch.cuda.reset_peak_memory_stats(device=device_id)
            torch.cuda.empty_cache()

        train_start_time = time.time()
        model.incremental_train(data_manager)
        train_time_stage_s = time.time() - train_start_time
        total_train_time += train_time_stage_s
        train_params_trainable_m, stage_trainable_param_names = _resolve_stage_trainable_stats(model)
        cumulative_trainable_param_names.update(stage_trainable_param_names)
        train_gpu_peak_mib = (
            torch.cuda.max_memory_allocated(device=device_id) / (1024 ** 2)
            if device_id is not None
            else 0.0
        )

        if device_id is not None:
            torch.cuda.reset_peak_memory_stats(device=device_id)
            torch.cuda.empty_cache()

        infer_num_samples = _resolve_infer_sample_count(model)
        infer_start_time = time.time()
        eval_result = model.eval_task()
        infer_time_stage_s = time.time() - infer_start_time
        total_test_time += infer_time_stage_s
        infer_gpu_peak_mib = (
            torch.cuda.max_memory_allocated(device=device_id) / (1024 ** 2)
            if device_id is not None
            else 0.0
        )

        cnn_accy, prompt_time_stage_s = _parse_eval_result(eval_result)
        total_prompt_time += prompt_time_stage_s
        infer_params_total_m = _resolve_inference_added_params_m(model, args)
        _append_stage_metrics(
            stage_metrics,
            task,
            model,
            args,
            train_params_trainable_m,
            train_time_stage_s,
            train_gpu_peak_mib,
            infer_params_total_m,
            infer_time_stage_s,
            infer_gpu_peak_mib,
            infer_num_samples,
            prompt_time_stage_s,
        )

        logging.info(
            "Task {} metrics => trainable {:.2f}M, inference added {:.2f}M, train GPU {:.2f}MiB, infer GPU {:.2f}MiB".format(
                task,
                train_params_trainable_m,
                infer_params_total_m,
                train_gpu_peak_mib,
                infer_gpu_peak_mib,
            )
        )

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
    print(
        "Average Accuracy (Top1): {}".format(
            round(sum(cnn_curve["top1"]) / len(cnn_curve["top1"]), 2)
        )
    )
    print("PD: {:.2f}".format(cnn_curve["top1"][0] - cnn_curve["top1"][-1]))
    print("Backward Transfer (BWT):", backward_transfer(np.array(cnn_matrix)))
    print("Forward Transfer (FWT):", forward_transfer(args["dataset"], np.array(cnn_matrix)))
    print("Total Train Time:", round(total_train_time, 2), "s")
    print("Total Inference Time:", round(total_test_time, 4), "s")
    print("Total Prompt Time:", round(total_prompt_time, 4), "s")
    if len(cnn_matrix) > 0:
        np_acctable = np.zeros([task + 1, task + 1])
        for idxx, line in enumerate(cnn_matrix):
            idxy = len(line)
            np_acctable[idxx, :idxy] = np.array(line)
        print("Accuracy Matrix (CNN):")
        print(np_acctable)

    print(f"{'=' * 80}\n")
    final_trainable_total_m = _resolve_final_trainable_total_m(model, cumulative_trainable_param_names)
    print_stage_metric_tables(stage_metrics, final_trainable_total_m)


def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus

    first_cuda = next((device for device in gpus if device.type == "cuda"), None)
    if first_cuda is not None:
        torch.cuda.set_device(first_cuda)
        # Force initialization on the selected device so subsequent CUDA memory
        # APIs do not fail with "did you call init?" and do not touch cuda:0.
        torch.empty(0, device=first_cuda)


def _set_random(seed=1, args=None):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        if args is not None:
            cuda_devices = [device for device in args["device"] if isinstance(device, torch.device) and device.type == "cuda"]
            if len(cuda_devices) > 1:
                torch.cuda.manual_seed_all(seed)
        else:
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
        hacc = 2 * old_acc * new_acc / (old_acc + new_acc)
    else:
        hacc = None
    return hacc, old_acc, new_acc


def backward_transfer(matrix):
    last_row_accs = matrix[-1, :-1]
    diagonal_accs = np.diag(matrix)[:-1]
    bwt_diffs = last_row_accs - diagonal_accs
    backward_transfer_value = np.mean(bwt_diffs)
    return np.round(backward_transfer_value, 2)


def forward_transfer(dataset, matrix):
    if dataset == "cub":
        rand_init_acc = np.array([86.51, 52.53, 65.04, 67.86, 74.62, 57.89, 71.82, 87.97, 68.6, 83.61, 85.37])
    elif dataset == "cifar224":
        rand_init_acc = np.array([77.5, 44.4, 51.8, 30.4, 56.8, 68.4, 44.0, 36.4, 41.4])
    elif dataset == "mini_imagenet":
        rand_init_acc = np.array([94.97, 90.6, 69.2, 81.2, 82.8, 73.4, 70.2, 90.6, 89.8])
    else:
        rand_init_acc = np.array([62.95, 28.08, 51.15, 38.64, 56.76, 50.93, 41.38, 61.25, 49.1, 43.37, 44.44])

    fwt_accs = np.diag(matrix, k=1)
    rand_init_acc_for_fwt = rand_init_acc[1:]

    if fwt_accs.shape[0] != rand_init_acc_for_fwt.shape[0]:
        return "FWT calculation failed: matrix length does not match."

    fwt_diffs = fwt_accs - rand_init_acc_for_fwt
    forward_transfer_value = np.mean(fwt_diffs)
    return np.round(forward_transfer_value, 2)
