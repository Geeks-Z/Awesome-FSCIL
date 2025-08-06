import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
import argparse
import gc
import torch
import time
import json

from transformers import CLIPProcessor

from CoLeLib.datasets import CIFAR100FSCIL, CUB200FSCIL, MiniImageNetFSCIL
from CoLeLib.training.strategies import CLIPPE, CLIPPEAblated

warnings.filterwarnings("ignore")

datasets = dict(
    cub200=dict(
        dataset=CUB200FSCIL, train_mb_size_base_class=4, train_epochs_base_class=6
    ),
    cifar100=dict(
        dataset=CIFAR100FSCIL, train_mb_size_base_class=32, train_epochs_base_class=5
    ),
    miniimagenet=dict(
        dataset=MiniImageNetFSCIL,
        train_mb_size_base_class=32,
        train_epochs_base_class=5,
    ),
)

parser = argparse.ArgumentParser(
    description="Train CPE-CLIP model on a few-shot class incremental learning task."
)
parser.add_argument(
    "--L_g", type=int, default=2, help="NuNumber of prompts to be used in the encoders"
)
parser.add_argument(
    "--deep_g",
    type=int,
    default=12,
    help="Number of layers of the encoders in which the prompts will be processed",
)
parser.add_argument(
    "--text_deep_replace_method",
    type=str,
    default="replace",
    choices=["replace", "accumalate", "accumulate_same"],
    help="Method to replace the text prompts in the encoders. Options: replace, accumulate, accumulate_same",
)
parser.add_argument(
    "--vision_deep_replace_method",
    type=str,
    default="accumulate",
    choices=["replace", "accumulate", "accumulate_same"],
    help="Method to replace the vision prompts in the encoders. Options: replace, accumulate, accumulate_same",
)
parser.add_argument(
    "--dataset_name",
    type=str,
    default="cifar100",
    choices=["cifar100", "cub200", "miniimagenet"],
    help="Name of the dataset to be used. Options: cifar100, cub200, miniimagenet",
)
parser.add_argument(
    "--n_runs",
    type=int,
    default=5,
    help="Number of runs to be executed",
    required=False,
)
parser.add_argument(
    "--seeds", type=int, nargs="+", help="Seeds to be used in the runs", required=False
)
parser.add_argument(
    "--ablation",
    type=str,
    choices=["no_accumulation", "no_regularization", "no_vision_prompts"],
)

args = parser.parse_args()

####

few_shot_examples = [5]
L_g = args.L_g
deep_g = args.deep_g
text_deep_replace_method = args.text_deep_replace_method
vision_deep_replace_method = args.vision_deep_replace_method
dataset_name = args.dataset_name
n_runs = args.n_runs
assert n_runs > 0, "Number of runs must be greater than 0."
# NOTE: Seeds used in the paper for 5 runs are: seeds = [42, 13, 50, 24, 69]
seeds = args.seeds if args.seeds else [s for s in range(42, 42 + n_runs)]
ablation = args.ablation
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
img_preprocess = CLIPProcessor.from_pretrained(
    "./models/clip-vit-base-patch16"
).feature_extractor

if __name__ == "__main__":

    # 全局时间统计
    all_runs_start_time = time.time()
    all_runs_stats = {
        "total_training_time": 0.0,
        "total_inference_time": 0.0,
        "runs_details": [],
    }

    for run in range(n_runs):
        print(f"\n{'='*50}")
        print(f"开始运行 Run {run+1}/{n_runs}")
        print(f"{'='*50}")

        # 单个run的时间统计
        run_start_time = time.time()
        run_stats = {
            "run_id": run + 1,
            "training_time": 0.0,
            "inference_time": 0.0,
            "experiences_stats": [],
        }

        exp_name = f"model_{dataset_name}_Lg{L_g}_Dg{deep_g}_TRM{text_deep_replace_method}_VRM{vision_deep_replace_method}_run{run+1}"

        Dataset = datasets[dataset_name]["dataset"]
        train_mb_size_base_class = datasets[dataset_name]["train_mb_size_base_class"]
        train_epochs_base_class = datasets[dataset_name]["train_epochs_base_class"]

        if ablation is None:
            strategy = CLIPPE(
                device="cuda",
                seed=seeds[run],
                L_g=L_g,
                deep_g=deep_g,
                text_deep_replace_method=text_deep_replace_method,
                vision_deep_replace_method=vision_deep_replace_method,
                regularization_method="balance",
                train_mb_size_base_class=train_mb_size_base_class,
                train_epochs_base_class=train_epochs_base_class,
                lr=0.00325,
                use_scheduler=True,
                json_file_name="temp_" + exp_name + ".json",
                eval_mb_size=64,
            )
        else:
            if ablation == "no_regularization":
                strategy = CLIPPE(
                    device="cuda",
                    seed=seeds[run],
                    L_g=L_g,
                    deep_g=deep_g,
                    text_deep_replace_method=text_deep_replace_method,
                    vision_deep_replace_method=vision_deep_replace_method,
                    regularization_method=None,
                    train_mb_size_base_class=train_mb_size_base_class,
                    train_epochs_base_class=train_epochs_base_class,
                    lr=0.00325,
                    use_scheduler=True,
                    json_file_name="temp_" + exp_name + ".json",
                    eval_mb_size=64,
                )
            elif ablation == "no_vision_prompts":
                strategy = CLIPPEAblated(
                    device="cuda",
                    seed=seeds[run],
                    L_g=L_g,
                    deep_g=deep_g,
                    text_deep_replace_method=text_deep_replace_method,
                    regularization_method="balance",
                    train_mb_size_base_class=train_mb_size_base_class,
                    train_epochs_base_class=train_epochs_base_class,
                    lr=0.00325,
                    use_scheduler=True,
                    json_file_name="temp_" + exp_name + ".json",
                    eval_mb_size=64,
                )
            elif ablation == "no_accumulation":
                strategy = CLIPPE(
                    device="cuda",
                    seed=seeds[run],
                    L_g=L_g,
                    deep_g=deep_g,
                    text_deep_replace_method="replace",
                    vision_deep_replace_method="replace",
                    regularization_method=None,
                    train_mb_size_base_class=train_mb_size_base_class,
                    train_epochs_base_class=train_epochs_base_class,
                    lr=0.00325,
                    use_scheduler=True,
                    json_file_name="temp_" + exp_name + ".json",
                    eval_mb_size=64,
                )
            else:
                raise ValueError("Invalid ablation option.")

        experiences = Dataset(transforms=img_preprocess)

        # 将时间统计信息传递给策略
        strategy.time_stats = run_stats
        strategy.train(experiences)

        # 计算本次run的总时间
        run_end_time = time.time()
        run_total_time = run_end_time - run_start_time

        # 打印本次run的统计信息
        print(f"\n{'='*50}")
        print(f"Run {run+1} 完成！")
        print(f"总训练时间: {run_stats['training_time']:.2f} 秒")
        print(f"总推理时间: {run_stats['inference_time']:.2f} 秒")
        print(f"Run总耗时: {run_total_time:.2f} 秒")
        print(f"{'='*50}")

        # 更新全局统计
        all_runs_stats["total_training_time"] += run_stats["training_time"]
        all_runs_stats["total_inference_time"] += run_stats["inference_time"]
        all_runs_stats["runs_details"].append(run_stats)

        torch.cuda.empty_cache()
        gc.collect()

        del strategy
        del experiences

    # 计算所有runs的总时间
    all_runs_end_time = time.time()
    all_runs_total_time = all_runs_end_time - all_runs_start_time

    # 打印最终统计信息
    print(f"\n{'='*70}")
    print(f"所有 {n_runs} 个 Runs 完成！最终统计:")
    print(f"{'='*70}")
    print(
        f"所有Runs总训练时间: {all_runs_stats['total_training_time']:.2f} 秒 ({all_runs_stats['total_training_time']/3600:.2f} 小时)"
    )
    print(
        f"所有Runs总推理时间: {all_runs_stats['total_inference_time']:.2f} 秒 ({all_runs_stats['total_inference_time']/3600:.2f} 小时)"
    )
    print(
        f"所有Runs总耗时: {all_runs_total_time:.2f} 秒 ({all_runs_total_time/3600:.2f} 小时)"
    )
    print(f"平均每个Run训练时间: {all_runs_stats['total_training_time']/n_runs:.2f} 秒")
    print(
        f"平均每个Run推理时间: {all_runs_stats['total_inference_time']/n_runs:.2f} 秒"
    )
    print(f"{'='*70}")

    # 保存详细统计信息到JSON文件
    time_stats_filename = (
        f"time_stats_{dataset_name}_Lg{L_g}_Dg{deep_g}_{n_runs}runs.json"
    )
    with open(time_stats_filename, "w", encoding="utf-8") as f:
        json.dump(all_runs_stats, f, indent=2, ensure_ascii=False)
    print(f"时间统计详情已保存到: {time_stats_filename}")
