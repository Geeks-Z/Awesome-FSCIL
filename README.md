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
## 📊 支持的 Backbone 对照表

| 方法            | ImageNet-1K Backbone               | ImageNet-21K Backbone                    | 状态         |
| --------------- | ---------------------------------- | ---------------------------------------- | ------------ |
| **L2P**         | `vit_base_patch16_224_l2p`         | `vit_base_patch16_224_in21k_l2p`         | ✅ 新增       |
| **DualPrompt**  | `vit_base_patch16_224_dualprompt`  | `vit_base_patch16_224_in21k_dualprompt`  | ✅ 已存在     |
| **CODA-Prompt** | `vit_base_patch16_224_coda_prompt` | `vit_base_patch16_224_in21k_coda_prompt` | ✅ 新增       |
| **Adapter**     | `pretrained_vit_b16_224_adapter`   | `pretrained_vit_b16_224_in21k_adapter`   | ✅ 已存在     |
| **EASE**        | `vit_base_patch16_224_ease`        | `vit_base_patch16_224_in21k_ease`        | ✅ 函数已存在 |
| **MOS**         | `vit_base_patch16_224_mos`         | `vit_base_patch16_224_in21k_mos`         | ✅ 函数已存在 |
| **ASP**         | `pretrained_vit_b16_224_vpt`       | `pretrained_vit_b16_224_in21k_vpt`       | ✅ 新增       |
| **SEC**         | `pretrained_vit_b16_224_vpt`       | `pretrained_vit_b16_224_in21k_vpt`       | ✅ 新增       |

---

## 

## 📝 Reproduced Results

### Evaluation protocol

以CIFAR B60 Inc5为例

![image-20250625151842896](https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/202506251518981.png)

1. `CNN` : 测试集在 $t$ 阶段模型 $M_t$ 上的准确率评估
   - `total`: $\frac{t阶段分类正确的样本数量}{t阶段所有样本数量}$ 
   $$
   A_t = \frac{\displaystyle \sum_{(\boldsymbol{x}_i, y_i) \in E_0 \cup E_1 \cup \ldots E_t} \left[ f(\boldsymbol{x}_i) = y_i \right]}{N_0^E + N_1^E + \ldots + N_t^E}
   $$
   where $N^E_{\tau}$ is the number of evaluation examples for session $\tau$, and $[\cdot]$ the indicator function.
   
   - `00-59`: $\frac{【0-59类别】分类正确的样本数量}{【0-59类别】阶段所有样本数量}$
1. `CNN top1 curve`: 每个阶段所有已见类别的分类准确率，即每个阶段的`total`
2. `Average Accuracy (CNN)`:  CNN top1 curve的平均值
3. `Forgetting (CNN)`: 所有增量阶段遗忘率的平均
   - 行代表不同的任务
   - 列显示每个任务在不同训练阶段的准确率
   - 每个任务的遗忘：每个任务在训练过程中的最高准确率 - 最后一列（最终准确率）

## 👨‍🏫 Acknowledgments