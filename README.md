## Awesome-FSCIL

<div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/logo.png"  /></div>

## 🎉 介绍

FSCIL: Few-shot Class-Incremental Learning/小样本类增量学习

## 🚀 Survey

| Title                                                        | Venue           | Year | Code | Code_Position |
| ------------------------------------------------------------ | --------------- | ---- | ---- | ------------- |
| [A survey on few-shot class-incremental learning](https://arxiv.org/abs/2304.08130) | Neural Networks | 2024 |      |               |
| [Few-shot Class-incremental Learning: A Survey](http://arxiv.org/abs/2308.06764) |                 |      |      |               |
|                                                              |                 |      |      |               |

## 🌟 Papers

| Title                                                        | Venue | Year | Type | Code         | Code_Position |
| ------------------------------------------------------------ | ----- | ---- | ---- | ------------ | ------------- |
| [MgSvF: Multi-Grained Slow versus Fast Framework for Few-Shot Class-Incremental Learning](https://ieeexplore.ieee.org/document/9645290/?arnumber=9645290) | TPAMI | 2024 |      | [Official]() | `📁  `         |
|                                                              |       |      |      |              |               |
|                                                              |       |      |      |              |               |
## 📝 Reproduced Results

### Evaluation protocol

以CIFAR B60 Inc5为例

![image-20250625151842896](https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/202506251518981.png)

1. `CNN` : 测试集在 $t$ 阶段模型 $M_t$ 上的准确率评估
   - `total`: $\frac{t阶段分类正确的样本数量}{t阶段所有样本数量}$ 
   - `00-59`: $\frac{【0-59类别】分类正确的样本数量}{【0-59类别】阶段所有样本数量}$
2. `CNN top1 curve`: 每个阶段所有已见类别的分类准确率，即每个阶段的`total`
3. `Average Accuracy (CNN)`:  CNN top1 curve的平均值
4. `Forgetting (CNN)`: 所有增量阶段遗忘率的平均
   - 行代表不同的任务
   - 列显示每个任务在不同训练阶段的准确率
   - 每个任务的遗忘：每个任务在训练过程中的最高准确率 - 最后一列（最终准确率）

## 👨‍🏫 Acknowledgments