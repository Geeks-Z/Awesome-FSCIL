#!/usr/bin/env python3
"""
示例运行脚本：展示如何使用修改后的训练代码进行时间统计

使用方法：
python run_with_timing.py --dataset_name cifar100 --n_runs 2 --L_g 2 --deep_g 12

这个脚本会：
1. 统计每个experience的训练时间和推理时间
2. 统计每个run的总训练时间和推理时间
3. 统计所有runs的总时间
4. 将详细统计保存到JSON文件中
"""

import subprocess
import sys
import os


def main():
    # 示例参数 - 你可以根据需要修改这些参数
    dataset_name = "cifar100"  # 可选: cifar100, cub200, miniimagenet
    n_runs = 2  # 运行次数，论文中使用5次
    L_g = 2  # prompt数量
    deep_g = 12  # 编码器层数
    text_deep_replace_method = "replace"
    vision_deep_replace_method = "accumulate"

    # 构建命令
    cmd = [
        sys.executable,
        "train.py",
        "--dataset_name",
        dataset_name,
        "--n_runs",
        str(n_runs),
        "--L_g",
        str(L_g),
        "--deep_g",
        str(deep_g),
        "--text_deep_replace_method",
        text_deep_replace_method,
        "--vision_deep_replace_method",
        vision_deep_replace_method,
    ]

    print("开始训练，命令:")
    print(" ".join(cmd))
    print("\n" + "=" * 70)

    # 运行训练
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n" + "=" * 70)
        print("训练完成！")

        # 检查生成的时间统计文件
        time_stats_file = (
            f"time_stats_{dataset_name}_Lg{L_g}_Dg{deep_g}_{n_runs}runs.json"
        )
        if os.path.exists(time_stats_file):
            print(f"时间统计文件已生成: {time_stats_file}")

            # 读取并显示简要统计
            import json

            with open(time_stats_file, "r", encoding="utf-8") as f:
                stats = json.load(f)

            print(f"\n简要统计:")
            print(f"总训练时间: {stats['total_training_time']:.2f} 秒")
            print(f"总推理时间: {stats['total_inference_time']:.2f} 秒")
            print(f"平均每run训练时间: {stats['total_training_time']/n_runs:.2f} 秒")
            print(f"平均每run推理时间: {stats['total_inference_time']/n_runs:.2f} 秒")

    except subprocess.CalledProcessError as e:
        print(f"训练过程中出现错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
