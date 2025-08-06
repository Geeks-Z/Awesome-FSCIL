# CPE-CLIP 时间统计功能使用指南

## 概述

此修改为CPE-CLIP训练代码添加了详细的时间统计功能，可以统计：

1. **每个Experience的时间**: 单独统计每个任务的训练时间和推理时间
2. **每个Run的时间**: 统计每次完整运行的总训练时间和推理时间
3. **全局时间统计**: 统计所有runs的总时间和平均时间
4. **详细报告**: 生成JSON格式的详细时间统计报告

## 修改内容

### 1. 主要修改文件

- `train.py`: 添加了全局时间统计逻辑
- `CoLeLib/training/templates/supervised.py`: 添加了experience级别的时间统计
- `run_with_timing.py`: 提供了简化的运行脚本
- `analyze_timing.py`: 提供了时间统计分析工具

### 2. 新增功能

#### 控制台输出
```
==================================================
开始运行 Run 1/3
==================================================
Experience 1 训练时间: 245.67 秒
总推理时间: 12.34 秒
Experience 2 训练时间: 123.45 秒
总推理时间: 8.76 秒
...
==================================================
Run 1 完成！
总训练时间: 369.12 秒
总推理时间: 21.10 秒
Run总耗时: 390.22 秒
==================================================
```

#### JSON统计文件
自动生成格式为 `time_stats_{dataset}_{参数}_{runs}runs.json` 的详细统计文件：

```json
{
  "total_training_time": 1107.36,
  "total_inference_time": 63.30,
  "runs_details": [
    {
      "run_id": 1,
      "training_time": 369.12,
      "inference_time": 21.10,
      "experiences_stats": [
        {
          "experience_id": 1,
          "training_time": 245.67,
          "inference_time": 12.34
        },
        ...
      ]
    },
    ...
  ]
}
```

## 使用方法

### 方法1: 直接运行训练脚本

```bash
python train.py --dataset_name cifar100 --n_runs 3 --L_g 2 --deep_g 12
```

训练完成后会自动：
- 在控制台显示详细时间统计
- 生成时间统计JSON文件

### 方法2: 使用简化运行脚本

```bash
python run_with_timing.py
```

这个脚本会使用预设参数运行训练，并自动分析生成的时间统计。

### 方法3: 分析已有的时间统计文件

```bash
# 基本分析
python analyze_timing.py time_stats_cifar100_Lg2_Dg12_3runs.json

# 生成图表分析
python analyze_timing.py time_stats_cifar100_Lg2_Dg12_3runs.json --plot
```

## 输出示例

### 控制台输出示例
```
======================================================================
所有 3 个 Runs 完成！最终统计:
======================================================================
所有Runs总训练时间: 1107.36 秒 (0.31 小时)
所有Runs总推理时间: 63.30 秒 (0.02 小时)
所有Runs总耗时: 1170.66 秒 (0.33 小时)
平均每个Run训练时间: 369.12 秒
平均每个Run推理时间: 21.10 秒
======================================================================
时间统计详情已保存到: time_stats_cifar100_Lg2_Dg12_3runs.json
```

### 分析报告示例
```
======================================================================
CPE-CLIP 训练时间统计分析报告
======================================================================

总体统计:
  运行次数: 3
  总训练时间: 1107.36 秒 (0.31 小时)
  总推理时间: 63.30 秒 (0.02 小时)
  总耗时: 1170.66 秒 (0.33 小时)

平均统计 (每个Run):
  平均训练时间: 369.12 秒 (6.15 分钟)
  平均推理时间: 21.10 秒 (0.35 分钟)
  平均总耗时: 390.22 秒 (6.50 分钟)

时间分布:
  训练时间占比: 94.6%
  推理时间占比: 5.4%
```

## 文件说明

### 生成的文件

1. **时间统计JSON文件**: `time_stats_{dataset}_Lg{L_g}_Dg{deep_g}_{n_runs}runs.json`
   - 包含所有详细的时间统计信息
   - 可用于后续分析和比较

2. **图表文件** (如果使用 `--plot` 选项):
   - `plots/timing_overview.png`: 总体时间分布图
   - `plots/experience_timing.png`: Experience级别时间统计图

### 依赖安装

如果需要生成图表，请安装matplotlib:

```bash
pip install matplotlib pandas
```

## 注意事项

1. **内存管理**: 代码在每个run结束后会自动清理GPU内存
2. **文件覆盖**: 相同参数的运行会覆盖之前的时间统计文件
3. **时间精度**: 所有时间统计精确到小数点后2位
4. **异常处理**: 如果训练过程中断，已完成的run时间统计仍会保存

## 自定义统计

如果需要添加更多统计指标，可以在以下位置修改：

1. `train.py` 中的 `run_stats` 字典 - 添加run级别统计
2. `supervised.py` 中的时间记录逻辑 - 添加experience级别统计
3. `analyze_timing.py` - 添加新的分析功能

## 示例运行命令

```bash
# CIFAR-100 数据集，运行5次
python train.py --dataset_name cifar100 --n_runs 5 --L_g 2 --deep_g 12

# CUB-200 数据集，运行3次
python train.py --dataset_name cub200 --n_runs 3 --L_g 2 --deep_g 12

# MiniImageNet 数据集，运行2次用于快速测试
python train.py --dataset_name miniimagenet --n_runs 2 --L_g 2 --deep_g 12
```

这些修改不会影响原有的训练逻辑和模型性能，只是添加了时间统计功能。
