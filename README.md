# Awesome-FSCIL

> Few-Shot Class-Incremental Learning (FSCIL) 与参数高效持续学习的统一复现仓库。

本仓库将多个 FSCIL / CIL 方法、补充实验配置、运行脚本与指标统计逻辑整理在一起，用于可复现的横向对比和后续扩展。实验结果汇总见 [`FSCIL_Results.xlsx`](FSCIL_Results.xlsx)。

## 📚 目录

- [仓库概览](#-仓库概览)
- [当前支持的方法](#-当前支持的方法)
- [SMP 复现](#-smp-复现)
- [指标统计说明](#-指标统计说明)
- [环境准备](#-环境准备)
- [数据集准备](#-数据集准备)

## 🔍 仓库概览

当前工作区主要关注以下内容：

- FSCIL 基线方法与基于 Prompt 的持续学习方法
- 多种 ViT 骨干网络，包括 ImageNet-1K、ImageNet-21K、DINO 以及部分 ViT-Large 变体
- 已统一到 task 级别的训练 / 推理指标输出
- 面向显存测试的 benchmark 配置

## 📦 当前支持的方法

下表根据当前工厂映射与配置目录整理，不以历史 README 为准。

| 方法族 | 位置 | 代表配置目录 | 典型入口 |
| --- | --- | --- | --- |
| SimpleCIL | `CIL_Model` | `CIL_Model/configs/simplecil/` | `CIL_Model/main.py` |
| Finetune | `CIL_Model` | `CIL_Model/configs/finetune/` | `CIL_Model/main.py` |
| L2P | `CIL_Model` | `CIL_Model/configs/l2p/` | `CIL_Model/main.py` |
| DualPrompt | `CIL_Model` | `CIL_Model/configs/dualprompt/` | `CIL_Model/main.py` |
| CODA-Prompt | `CIL_Model` | `CIL_Model/configs/coda_prompt/` | `CIL_Model/main.py` |
| LAE | `CIL_Model` | `CIL_Model/configs/lae/` | `CIL_Model/main.py` |
| Adapter / APER 风格配置 | `CIL_Model` | `CIL_Model/configs/aper/` | `CIL_Model/main.py` |
| ASP | `FSCIL-ASP-main` | `FSCIL-ASP-main/configs/` | `FSCIL-ASP-main/main.py` |
| SEC-Prompt | `SEC-Prompt-main` | `SEC-Prompt-main/configs/` | `SEC-Prompt-main/main.py` |
| SMP | `SMP-main` | `SMP-main/configs/` | `SMP-main/main.py` |
| InfLoRA / 变体 | `CIL_Model/InfLoRA-main` | `CIL_Model/InfLoRA-main/configs/` | `CIL_Model/InfLoRA-main/main.py` |
| SD-LoRA | `CIL_Model/SD-Lora-CL-main` | `CIL_Model/SD-Lora-CL-main/configs/` | `CIL_Model/SD-Lora-CL-main/main.py` |

## 🧪 SMP 复现

[`SMP-main`](SMP-main) 基于 [SMP 上游实现](https://github.com/beiyan1911/SMP) `4ab7b08` 接入，保留独立目录、开源许可证与原始配置，并补充了以下内容：

- 与工作簿前四个 sheet 一致的实验顺序：`IN1K-NoShuffle1993` → `IN1K-Shuffle1993` → `IN1K-Shuffle2025` → `IN1K-Shuffle42`。
- 每组按 `CUB200` → `CIFAR100` → `ImageNet-R` → `miniImageNet` 运行，统一使用 ImageNet-1K 预训练的 ViT-B/16。
- 补充 SMP 上游未提供的 miniImageNet 数据加载与配置。
- 外层输出按 `logs/<sheet>/SMP-IN1K-<Shuffle>-<seed>-A800.out` 归档，细粒度日志保存在 `SMP-main/logs/`。队列配置默认设置 `save_checkpoints: false`；需要保存模型时可在对应 JSON 中显式开启。

当前复现实验已完成 8 / 16 组（50%）；逐项状态、已完成指标和续跑命令见 [`SMP-main/EXPERIMENT_STATUS.md`](SMP-main/EXPERIMENT_STATUS.md)。

运行单个 sheet：

```bash
cd SMP-main
CUDA_VISIBLE_DEVICES=0 ./train_smp.sh IN1K-NoShuffle1993
```

按工作簿顺序运行全部 SMP 实验：

```bash
cd SMP-main
./submit_smp_gpu0.sh
```

## 📏 指标统计说明

当前工作区中的若干 trainer 已统一为“训练指标”和“推理指标”分开统计，并按 task 输出表格。

### 指标定义

#### 训练指标

- `Trainable(M)`：当前 task 中真实参与训练的参数量。
- `GPU(MiB)`：训练阶段峰值显存。
- `Time/Task(s)`：当前 task 的训练总时间。
- `Epochs`：当前 task 使用的 epoch 数。
- `Time/Epoch(s)`：`Time/Task(s) / Epochs`。
- `Final`：完整持续学习过程结束后，最终参与训练的总参数量。

#### 推理指标

- `Added(M)`：相对于一个 frozen backbone 的最终部署新增参数量。
- `GPU(MiB)`：推理阶段峰值显存。
- `Time/Task(s)`：当前 task 的完整推理时间。
- `Latency(ms/img)`：`Time/Task(s) / 当前 task 推理样本数`。
- `Samples`：当前 task 推理样本数。
- `Prompt(s)`：与 prompt 选择相关的时间。对于使用辅助 original-backbone branch 做 prompt select 的方法，这部分时间被计入 `Prompt(s)`。

### Future Evaluation 统一口径

当前 future evaluation 已统一为两条 loader：

- `future_train_loader`：从未来类别的训练集构建，使用 `mode="test"`，必要时保留 `kshot`。仅用于 future head / prototype 构造。
- `future_test_loader`：从未来类别的测试集构建，使用 `mode="test"`。仅用于 future evaluation 的正式统计。

这样统一后，可以避免：

- 用训练集直接做未来类报告
- 在评测时混入训练增强
- 用测试集样本构造 future head 造成泄漏

### 各方法的统计策略

下表给出当前仓库中各方法族的训练 / 推理统计策略。

| 方法族 | 训练阶段 `Trainable(M)` 统计策略 | 推理阶段 `Added(M)` 统计策略 | Future Evaluation 策略 |
| --- | --- | --- | --- |
| L2P | 统计当前 task 真正参与训练的新增 prompt / classifier 参数，排除 `original_backbone` 与 `future_head` | 只统计新增的 prompt pool 与 classifier head，不计 frozen backbone 与 `original_backbone` | 用 `future_train_loader` 构造 future prototypes，用 `future_test_loader` 报告 |
| DualPrompt | 与 L2P 相同 | 只统计新增的 g-prompt、e-prompt、prompt key 与 classifier head | 用 `future_train_loader` 构造 future prototypes，用 `future_test_loader` 报告 |
| CODA-Prompt | 统计当前 task 参与训练的 prompt / classifier 参数，排除 `future_head` | 只统计新增的 prompt 模块与 classifier head | 用 `future_train_loader` 构造 future prototypes，用 `future_test_loader` 报告 |
| Finetune | 统计当前 task 实际更新的参数，排除 `future_head` | 只统计相对 frozen backbone 新增的 classifier head | 用 `future_train_loader` 构造 future prototypes，用 `future_test_loader` 报告 |
| LAE | 统计当前 task 实际参与训练的 PET / classifier 参数 | 只统计新增 PET 模块与 classifier head，排除 EMA 副本与 `future_head` | 用 `future_train_loader` 构造 future prototypes，用 `future_test_loader` 报告 |
| SimpleCIL | 统计当前 task 中参与训练 / 原型更新的参数，排除 `future_head` | 只统计相对 frozen backbone 新增的 classifier head | 用 `future_train_loader` 构造 future prototypes，用 `future_test_loader` 报告 |
| ASP | 统计当前 task 真实参与训练的参数，排除仅用于 future evaluation 的 `future_head` | 只统计 TIP、prompt encoder MLP 与 classifier head，排除冻结辅助 backbone | 用 `future_train_loader` 构造 future prototypes，用 `future_test_loader` 报告 |
| SEC-Prompt | 统计当前 task 真实参与训练的 prompt / classifier 参数，排除 `future_head` | 只统计 TSP、RSP 与 classifier head | 用 `future_train_loader` 构造 future prototypes，用 `future_test_loader` 报告 |
| InfLoRA / 变体 | 统计当前 task 激活的 LoRA 分支与分类器参数，排除 `future_head` | 统计最终部署时相对 frozen backbone 保留的 LoRA 分支与 classifier pool | 用 `future_train_loader` 构造 future prototypes，用 `future_test_loader` 报告 |
| SD-LoRA | 统计当前 task 激活的 LoRA 与分类器参数，排除 `future_head` | 统计最终部署时相对 frozen backbone 保留的 LoRA 分支与 classifier head | 用 `future_train_loader` 构造 future prototypes，用 `future_test_loader` 报告 |

## ⚙️ 环境准备

1. torch 2.0.1
2. torchvision 0.15.2
3. timm 0.6.12
4. tqdm
5. numpy
6. scipy
7. easydict

SMP 上游的参考环境为 Python 3.10、PyTorch 2.1.0、torchvision 0.16.0 和 timm 0.6.7；当前集成也兼容本项目使用的 timm 0.6.x 环境。完整依赖见 [`SMP-main/install.txt`](SMP-main/install.txt)。

## 🗃️ 数据集准备

当前服务器统一数据根目录为：

```text
/public/home/hanlida/Dr.1/Dataset
```

SMP 配置通过 `data_root` 指定该目录，也可用 `FSCIL_DATA_ROOT` 环境变量覆盖。其他方法的部分数据加载器仍在 `utils/data.py` 中使用绝对路径，例如：

- [FSCIL-ASP-main/utils/data.py](FSCIL-ASP-main/utils/data.py)
- [SEC-Prompt-main/utils/data.py](SEC-Prompt-main/utils/data.py)
- [CIL_Model/utils/data.py](CIL_Model/utils/data.py)
- [CIL_Model/InfLoRA-main/utils/data.py](CIL_Model/InfLoRA-main/utils/data.py)
- [CIL_Model/SD-Lora-CL-main/utils/data.py](CIL_Model/SD-Lora-CL-main/utils/data.py)

当前代码中涉及的数据集包括：

- **CIFAR100**: will be automatically downloaded by the code.
- **CUB200**: Google Drive: [link](https://drive.google.com/file/d/1Swpje08SXizLX1QCJzNayIMgvHU1gtVe/view?usp=sharing)
- **miniImageNet**: Google Drive: [link](https://drive.google.com/file/d/1Nq7J-y17cNRDs7bG2h_PTbX1vBYKJ16u/view?usp=sharing)
- **ImageNet-R**: Google Drive: [link](https://drive.google.com/file/d/1SG4TbiL8_DooekztyCVK8mPmfhMo8fkR/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EU4jyLL29CtBsZkB6y-JSbgBzWF5YHhBAUz1Qw8qM2954A?e=hlWpNW)

When using a custom dataset root for a legacy method, update the corresponding mapping:
```python
# In `utils/data.py`
def download_data(self):
    train_dir = '[YOUR_DATA_PATH]/train/'
    test_dir = '[YOUR_DATA_PATH]/val/'
```
