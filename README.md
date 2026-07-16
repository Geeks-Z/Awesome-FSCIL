# 🚀 Awesome-FSCIL

<div align="center">
  <strong>Reproductions for Few-Shot Class-Incremental Learning (FSCIL) and parameter-efficient continual learning.</strong>
</div>
<p></p>
<div align="center">
  <img src="https://visitor-badge.laobi.icu/badge?page_id=hongwei-zhao.Awesome-FSCIL&left_color=green&right_color=red" />
  <a href="https://github.com/hongwei-zhao/Awesome-FSCIL/commits/main"><img src="https://img.shields.io/github/last-commit/hongwei-zhao/Awesome-FSCIL" /></a>
  <a href="https://github.com/hongwei-zhao/Awesome-FSCIL"><img src="https://img.shields.io/github/repo-size/hongwei-zhao/Awesome-FSCIL" /></a>
</div>

## 🎉 Introduction

- **Few-Shot Class-Incremental Learning (FSCIL)** studies how models learn new classes from only a few examples while retaining performance on previously learned classes.
- **This repository collects** FSCIL/CIL baselines and prompt-, adapter-, and LoRA-based methods, along with their code, configurations, run scripts, and reproduced results.
- **Unified settings** include ImageNet-1K, ImageNet-21K, DINO, and selected ViT-Large initializations. Training and inference metrics are recorded at the task level.
- **Results** are summarized in [`FSCIL_Results.xlsx`](FSCIL_Results.xlsx). Detailed SMP progress and completed accuracy curves are available in [`SMP-main/EXPERIMENT_STATUS.md`](SMP-main/EXPERIMENT_STATUS.md).

---

## 🧭 Methods Reproduced

- `FineTune`: A baseline that directly fine-tunes the model on every incremental task.
- `SimpleCIL / APER`: *Revisiting Class-Incremental Learning with Pre-Trained Models: Generalizability and Adaptivity are All You Need*. **IJCV 2024** [[paper](https://arxiv.org/abs/2303.07338)]
- `L2P`: *Learning to Prompt for Continual Learning*. **CVPR 2022** [[paper](https://arxiv.org/abs/2112.08654)]
- `DualPrompt`: *DualPrompt: Complementary Prompting for Rehearsal-Free Continual Learning*. **ECCV 2022** [[paper](https://arxiv.org/abs/2204.04799)]
- `CODA-Prompt`: *CODA-Prompt: COntinual Decomposed Attention-based Prompting for Rehearsal-Free Continual Learning*. **CVPR 2023** [[paper](https://arxiv.org/abs/2211.13218)]
- `LAE`: *A Unified Continual Learning Framework with General Parameter-Efficient Tuning*. **ICCV 2023** [[paper](https://arxiv.org/abs/2303.10070)]
- `EASE`: *Expandable Subspace Ensemble for Pre-Trained Model-Based Class-Incremental Learning*. **CVPR 2024** [[paper](https://openaccess.thecvf.com/content/CVPR2024/html/Zhou_Expandable_Subspace_Ensemble_for_Pre-Trained_Model-Based_Class-Incremental_Learning_CVPR_2024_paper.html)]
- `ASP`: *Few-Shot Class Incremental Learning with Attention-Aware Self-Adaptive Prompt*. [[paper](https://arxiv.org/abs/2403.09857)] [[code](https://github.com/dawnliu35/fscil-asp)]
- `SEC-Prompt`: *SEC-Prompt:SEmantic Complementary Prompting for Few-Shot Class-Incremental Learning*. **CVPR 2025** [[paper](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_SEC-PromptSEmantic_Complementary_Prompting_for_Few-Shot_Class-Incremental_Learning_CVPR_2025_paper.html)]
- `SMP`: *Sculpting Margin Penalty: Intra-Task Adapter Merging and Classifier Calibration for Few-Shot Class-Incremental Learning*. [[paper](https://arxiv.org/abs/2508.05094)] [[code](https://github.com/beiyan1911/SMP)]
- `InfLoRA`: *InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning*. **CVPR 2024** [[paper](https://arxiv.org/abs/2404.00228)]
- `SD-LoRA`: *SD-LoRA: Scalable Decoupled Low-Rank Adaptation for Class Incremental Learning*. **ICLR 2025** [[paper](https://arxiv.org/abs/2501.13198)]

| Method | Source directory | Configuration directory |
| --- | --- | --- |
| FineTune, SimpleCIL, L2P, DualPrompt, CODA-Prompt, LAE, EASE, APER | [`CIL_Model`](CIL_Model) | `CIL_Model/configs/<method>/` |
| ASP | [`FSCIL-ASP-main`](FSCIL-ASP-main) | `FSCIL-ASP-main/configs/` |
| SEC-Prompt | [`SEC-Prompt-main`](SEC-Prompt-main) | `SEC-Prompt-main/configs/` |
| SMP | [`SMP-main`](SMP-main) | `SMP-main/configs/<setting>/` |
| InfLoRA | [`CIL_Model/InfLoRA-main`](CIL_Model/InfLoRA-main) | `CIL_Model/InfLoRA-main/configs/` |
| SD-LoRA | [`CIL_Model/SD-Lora-CL-main`](CIL_Model/SD-Lora-CL-main) | `CIL_Model/SD-Lora-CL-main/configs/` |

---

## ☀️ How to Use

### 🗒️ Clone

```bash
git clone https://github.com/hongwei-zhao/Awesome-FSCIL.git
cd Awesome-FSCIL
```

### 🛠️ Dependencies

The primary framework uses the following dependencies:

1. [PyTorch 2.0.1](https://github.com/pytorch/pytorch)
2. [torchvision 0.15.2](https://github.com/pytorch/vision)
3. [timm 0.6.12](https://github.com/huggingface/pytorch-image-models)
4. [tqdm](https://github.com/tqdm/tqdm)
5. [numpy](https://github.com/numpy/numpy)
6. [scipy](https://github.com/scipy/scipy)
7. [easydict](https://github.com/makinacorpus/easydict)

Each reproduction directory retains its upstream environment. SMP recommends Python 3.10, PyTorch 2.1.0, torchvision 0.16.0, and timm 0.6.7; see [`SMP-main/install.txt`](SMP-main/install.txt) for the full list. ASP-specific requirements are in [`FSCIL-ASP-main/requirements.txt`](FSCIL-ASP-main/requirements.txt).

### 🚀 Run an Experiment

1. Select or edit a JSON configuration in the target method's `configs/` directory. Common fields are:

   - **dataset**: Dataset identifier, such as `cifar224`, `cub200`, `miniimagenet`, or `imagenet-r`.
   - **init_cls**: Number of classes in the base session.
   - **increment**: Number of classes introduced in each subsequent session.
   - **kshot**: Number of available examples per new class.
   - **backbone_type**: Pre-trained ViT backbone and initialization variant.
   - **seed**, **shuffle**, and **device**: Class order and execution-device settings.

2. Run L2P in `CIL_Model`, for example:

   ```bash
   cd CIL_Model
   python main.py --config=./configs/l2p/l2p_cifar_B60_Inc5.json
   ```

3. ASP and SEC-Prompt use the entry point in their own directories:

   ```bash
   cd FSCIL-ASP-main
   python main.py --config=./configs/asp_cifar_B60_Inc5.json

   cd ../SEC-Prompt-main
   python main.py --config=./configs/SEC-Prompt_cifar_B60_Inc5.json
   ```

### 🧪 SMP Reproduction

SMP includes miniImageNet support and organizes experiments by four class orders: `IN1K-NoShuffle1993`, `IN1K-Shuffle1993`, `IN1K-Shuffle2025`, and `IN1K-Shuffle42`. Each setting runs `CUB200`, `CIFAR100`, `ImageNet-R`, and `miniImageNet` in that order.

Run one setting:

```bash
cd SMP-main
CUDA_VISIBLE_DEVICES=0 ./train_smp.sh IN1K-NoShuffle1993
```

Submit all SMP experiments in workbook order:

```bash
cd SMP-main
./submit_smp_gpu0.sh
```

The default configurations do not save checkpoints (`save_checkpoints: false`). Enable the field explicitly in the corresponding JSON file when checkpoints are required.

---

## 📊 Results and Evaluation

[`FSCIL_Results.xlsx`](FSCIL_Results.xlsx) summarizes results across methods, datasets, and class orders. SMP has completed 8 of the 16 planned runs; see [`SMP-main/EXPERIMENT_STATUS.md`](SMP-main/EXPERIMENT_STATUS.md) for the authoritative status, final Top-1 curves, and resume commands.

Training and inference metrics are reported separately for every task:

- **Training**: `Trainable(M)`, `GPU(MiB)`, `Time/Task(s)`, `Epochs`, `Time/Epoch(s)`, and final trainable parameters (`Final`).
- **Inference**: Parameters added relative to a frozen backbone (`Added(M)`), peak GPU memory, inference time, `Latency(ms/img)`, sample count, and prompt-selection time.

Future evaluation uses independent loaders: `future_train_loader` only constructs the future head/prototypes, while `future_test_loader` is only used for formal reporting. This prevents test-set leakage and training augmentation from entering evaluation.

---

## 📎 Datasets

| Dataset | Classes | Usage | Download |
| --- | ---: | --- | --- |
| CIFAR100 | 100 | Downloaded automatically by the code | — |
| CUB200 | 200 | FSCIL benchmark | [Google Drive](https://drive.google.com/file/d/1Swpje08SXizLX1QCJzNayIMgvHU1gtVe/view?usp=sharing) |
| miniImageNet | 100 | FSCIL benchmark | [Google Drive](https://drive.google.com/file/d/1Nq7J-y17cNRDs7bG2h_PTbX1vBYKJ16u/view?usp=sharing) |
| ImageNet-R | 200 | FSCIL benchmark | [Google Drive](https://drive.google.com/file/d/1SG4TbiL8_DooekztyCVK8mPmfhMo8fkR/view?usp=sharing) / [OneDrive](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EU4jyLL29CtBsZkB6y-JSbgBzWF5YHhBAUz1Qw8qM2954A?e=hlWpNW) |

SMP reads its dataset root from `data_root` in the configuration and allows the `FSCIL_DATA_ROOT` environment variable to override it. Some legacy implementations still use absolute paths in their own `utils/data.py`; update the corresponding `train_dir` and `test_dir` mappings when using a custom root.

```python
def download_data(self):
    train_dir = '[YOUR_DATA_PATH]/train/'
    test_dir = '[YOUR_DATA_PATH]/val/'
```

---

## 🎓 Acknowledgments

- [SMP](https://github.com/beiyan1911/SMP)
- [FSCIL-ASP](https://github.com/dawnliu35/fscil-asp)
- [PyCIL](https://github.com/G-U-N/PyCIL)
- [Awesome-Incremental-Learning](https://github.com/xialeiliu/Awesome-Incremental-Learning)
