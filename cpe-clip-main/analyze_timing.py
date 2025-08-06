#!/usr/bin/env python3
"""
时间统计分析工具

用于分析训练过程中生成的时间统计文件，提供详细的时间分析报告

使用方法：
python analyze_timing.py time_stats_cifar100_Lg2_Dg12_2runs.json
"""

import json
import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def analyze_timing_stats(json_file_path):
    """分析时间统计JSON文件"""

    if not Path(json_file_path).exists():
        print(f"错误: 文件 {json_file_path} 不存在")
        return

    # 读取统计数据
    with open(json_file_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    print("=" * 70)
    print("CPE-CLIP 训练时间统计分析报告")
    print("=" * 70)

    # 总体统计
    total_training = stats["total_training_time"]
    total_inference = stats["total_inference_time"]
    n_runs = len(stats["runs_details"])

    print(f"\n总体统计:")
    print(f"  运行次数: {n_runs}")
    print(f"  总训练时间: {total_training:.2f} 秒 ({total_training/3600:.2f} 小时)")
    print(f"  总推理时间: {total_inference:.2f} 秒 ({total_inference/3600:.2f} 小时)")
    print(
        f"  总耗时: {total_training + total_inference:.2f} 秒 ({(total_training + total_inference)/3600:.2f} 小时)"
    )

    # 平均统计
    avg_training = total_training / n_runs
    avg_inference = total_inference / n_runs

    print(f"\n平均统计 (每个Run):")
    print(f"  平均训练时间: {avg_training:.2f} 秒 ({avg_training/60:.2f} 分钟)")
    print(f"  平均推理时间: {avg_inference:.2f} 秒 ({avg_inference/60:.2f} 分钟)")
    print(
        f"  平均总耗时: {avg_training + avg_inference:.2f} 秒 ({(avg_training + avg_inference)/60:.2f} 分钟)"
    )

    # 每个Run的详细统计
    print(f"\n每个Run的详细统计:")
    print(
        f"{'Run':<4} {'训练时间(秒)':<12} {'推理时间(秒)':<12} {'总时间(秒)':<10} {'Experience数':<12}"
    )
    print("-" * 60)

    for run_detail in stats["runs_details"]:
        run_id = run_detail["run_id"]
        train_time = run_detail["training_time"]
        infer_time = run_detail["inference_time"]
        total_time = train_time + infer_time
        n_exp = len(run_detail["experiences_stats"])

        print(
            f"{run_id:<4} {train_time:<12.2f} {infer_time:<12.2f} {total_time:<10.2f} {n_exp:<12}"
        )

    # Experience级别的统计 (以第一个run为例)
    if stats["runs_details"] and stats["runs_details"][0]["experiences_stats"]:
        print(f"\nExperience级别统计 (以Run 1为例):")
        print(
            f"{'Exp':<4} {'训练时间(秒)':<12} {'推理时间(秒)':<12} {'总时间(秒)':<10}"
        )
        print("-" * 40)

        for exp_stat in stats["runs_details"][0]["experiences_stats"]:
            exp_id = exp_stat["experience_id"]
            train_time = exp_stat["training_time"]
            infer_time = exp_stat["inference_time"]
            total_time = train_time + infer_time

            print(
                f"{exp_id:<4} {train_time:<12.2f} {infer_time:<12.2f} {total_time:<10.2f}"
            )

    # 时间分布分析
    training_percentage = (total_training / (total_training + total_inference)) * 100
    inference_percentage = (total_inference / (total_training + total_inference)) * 100

    print(f"\n时间分布:")
    print(f"  训练时间占比: {training_percentage:.1f}%")
    print(f"  推理时间占比: {inference_percentage:.1f}%")

    return stats


def create_timing_plots(stats, output_dir="plots"):
    """创建时间统计图表"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        Path(output_dir).mkdir(exist_ok=True)

        # 1. 训练vs推理时间对比图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # 饼图: 总时间分布
        total_training = stats["total_training_time"]
        total_inference = stats["total_inference_time"]

        ax1.pie(
            [total_training, total_inference],
            labels=["训练时间", "推理时间"],
            autopct="%1.1f%%",
            startangle=90,
        )
        ax1.set_title("总时间分布")

        # 柱状图: 每个run的时间
        runs = [f"Run {r['run_id']}" for r in stats["runs_details"]]
        train_times = [r["training_time"] for r in stats["runs_details"]]
        infer_times = [r["inference_time"] for r in stats["runs_details"]]

        x = np.arange(len(runs))
        width = 0.35

        ax2.bar(x - width / 2, train_times, width, label="训练时间", alpha=0.8)
        ax2.bar(x + width / 2, infer_times, width, label="推理时间", alpha=0.8)

        ax2.set_xlabel("Run")
        ax2.set_ylabel("时间 (秒)")
        ax2.set_title("每个Run的时间统计")
        ax2.set_xticks(x)
        ax2.set_xticklabels(runs)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(f"{output_dir}/timing_overview.png", dpi=300, bbox_inches="tight")
        print(f"图表已保存到 {output_dir}/timing_overview.png")

        # 2. Experience级别的时间图 (以第一个run为例)
        if stats["runs_details"] and stats["runs_details"][0]["experiences_stats"]:
            fig, ax = plt.subplots(figsize=(10, 6))

            exp_stats = stats["runs_details"][0]["experiences_stats"]
            exp_ids = [f"Exp {e['experience_id']}" for e in exp_stats]
            exp_train_times = [e["training_time"] for e in exp_stats]
            exp_infer_times = [e["inference_time"] for e in exp_stats]

            x = np.arange(len(exp_ids))

            ax.bar(x - width / 2, exp_train_times, width, label="训练时间", alpha=0.8)
            ax.bar(x + width / 2, exp_infer_times, width, label="推理时间", alpha=0.8)

            ax.set_xlabel("Experience")
            ax.set_ylabel("时间 (秒)")
            ax.set_title("每个Experience的时间统计 (Run 1)")
            ax.set_xticks(x)
            ax.set_xticklabels(exp_ids)
            ax.legend()

            plt.tight_layout()
            plt.savefig(
                f"{output_dir}/experience_timing.png", dpi=300, bbox_inches="tight"
            )
            print(f"图表已保存到 {output_dir}/experience_timing.png")

    except ImportError:
        print("注意: matplotlib未安装，跳过图表生成")
    except Exception as e:
        print(f"生成图表时出错: {e}")


def main():
    parser = argparse.ArgumentParser(description="分析CPE-CLIP训练时间统计")
    parser.add_argument("json_file", help="时间统计JSON文件路径")
    parser.add_argument("--plot", action="store_true", help="生成时间统计图表")
    parser.add_argument("--output_dir", default="plots", help="图表输出目录")

    args = parser.parse_args()

    # 分析统计数据
    stats = analyze_timing_stats(args.json_file)

    # 生成图表
    if args.plot and stats:
        create_timing_plots(stats, args.output_dir)


if __name__ == "__main__":
    main()
