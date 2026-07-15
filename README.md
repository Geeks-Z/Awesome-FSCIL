# Awesome-FSCIL

> 面向 Few-Shot Class-Incremental Learning (FSCIL) 与参数高效持续学习方法的代码仓库。

本仓库不将多个 FSCIL / CIL 子项目、补充配置、benchmark 配置以及统一后的指标统计逻辑整理在一起，便于横向对比、实验验证与后续扩展。

## 📚 目录

- [仓库概览](#-仓库概览)
- [当前支持的方法](#-当前支持的方法)
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
| InfLoRA / 变体 | `CIL_Model/InfLoRA-main` | `CIL_Model/InfLoRA-main/configs/` | `CIL_Model/InfLoRA-main/main.py` |
| SD-LoRA | `CIL_Model/SD-Lora-CL-main` | `CIL_Model/SD-Lora-CL-main/configs/` | `CIL_Model/SD-Lora-CL-main/main.py` |

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

## 🗃️ 数据集准备

当前代码中的多数数据加载器仍然在 `utils/data.py` 内使用了硬编码的绝对路径，例如：

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

When utilizing custom or downloaded datasets, specify the absolute folder trajectory mapping manually internally:
```python
# In `utils/data.py`
def download_data(self):
    train_dir = '[YOUR_DATA_PATH]/train/'
    test_dir = '[YOUR_DATA_PATH]/val/'
```