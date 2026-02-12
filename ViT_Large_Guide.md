# ViT-Large (vit_large_patch16_224) 模型支持指南

本文档说明如何在 CIL_Model、FSCIL-ASP、SEC-Prompt、InfLoRA、SD-LoRA 中使用 ViT-Large 模型。

## 模型架构

| 参数 | ViT-Base/16 | ViT-Large/16 |
|------|-------------|--------------|
| `embed_dim` | 768 | 1024 |
| `depth` | 12 | 24 |
| `num_heads` | 12 | 16 |
| `patch_size` | 16 | 16 |
| `params` | 86M | 307M |

> [!IMPORTANT]
> **显存需求**：ViT-Large 参数量约为 ViT-Base 的 4 倍（307M vs 86M），建议使用 24GB+ GPU，batch_size 建议减半。

---

## 已支持的方法

| 项目 | 方法 | backbone_type |
|------|------|---------------|
| CIL_Model | Adapter | `pretrained_vit_large_patch16_224_adapter` |
| CIL_Model | L2P | `vit_large_patch16_224_l2p` |
| CIL_Model | DualPrompt | `vit_large_patch16_224_dualprompt` |
| CIL_Model | Coda_Prompt | `vit_large_patch16_224_coda_prompt` |
| CIL_Model | EASE | `vit_large_patch16_224_ease` |
| InfLoRA | InfLoRA | 通过 `embd_dim=1024, num_heads=16` 自动选择 |
| SD-LoRA | SD-LoRA | `vit_large_patch16_224` |
| FSCIL-ASP | ASP | `pretrained_vit_large_patch16_224_vpt` |
| SEC-Prompt | SEC-Prompt | `pretrained_vit_large_patch16_224_vpt` |

---

## 已修改的文件

### CIL_Model/backbone/

| 文件 | 修改内容 |
|------|----------|
| [vit_adapter.py](CIL_Model/backbone/vit_adapter.py) | 新增 `vit_large_patch16_224_adapter()` |
| [vit_ease.py](CIL_Model/backbone/vit_ease.py) | 新增 `vit_large_patch16_224_ease()` |
| [vit_l2p.py](CIL_Model/backbone/vit_l2p.py) | 已有 `vit_large_patch16_224_l2p()` |
| [vit_dualprompt.py](CIL_Model/backbone/vit_dualprompt.py) | 已有 `vit_large_patch16_224_dualprompt()` |
| [vit_coda_promtpt.py](CIL_Model/backbone/vit_coda_promtpt.py) | 已有 `vit_large_patch16_224_coda_prompt()` |

### CIL_Model/utils/inc_net.py

添加各方法的 ViT-Large 分支，动态设置 `d_model=1024` 和 `out_dim=1024`

### InfLoRA

| 文件 | 修改内容 |
|------|----------|
| [sinet_inflora.py](CIL_Model/InfLoRA-main/models/sinet_inflora.py) | 根据 `embd_dim` 动态选择 ViT-Base/Large 模型 |

```python
# 关键代码变更
if embd_dim == 1024:
    # ViT-Large: embed_dim=1024, depth=24, num_heads=16
    model_kwargs = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, ...)
    self.image_encoder = _create_vision_transformer('vit_large_patch16_224', pretrained=True, **model_kwargs)
else:
    # ViT-Base
    self.image_encoder = _create_vision_transformer('vit_base_patch16_224', pretrained=True, **model_kwargs)
```

### SD-LoRA

| 文件 | 修改内容 |
|------|----------|
| [inc_net.py](CIL_Model/SD-Lora-CL-main/utils/inc_net.py) | 添加 `vit_large_patch16_224` 分支 (out_dim=1024) |

### FSCIL-ASP & SEC-Prompt

| 文件 | 修改内容 |
|------|----------|
| asp_backbone.py / vpt_backbone.py | 动态架构参数 + SimpleVitNet 动态维度 |
| zoo.py (SEC-Prompt) | 动态 e_layers（前 1/3 层） |
| inc_net.py | 添加 ViT-Large 分支 |

---

## 配置文件列表 (_L16 命名)

### CIL_Model 方法

| 方法 | 目录 | 配置文件 |
|------|------|----------|
| L2P | configs/l2p/ | l2p_{cub,cifar,mini,inr}_*_L16.json |
| DualPrompt | configs/dualprompt/ | dualprompt_{cub,cifar,mini,inr}_*_L16.json |
| Coda_Prompt | configs/coda_prompt/ | coda_prompt_{cub,cifar,mini,inr}_*_L16.json |
| EASE | configs/ease/ | ease_{cub,cifar,mini,inr}_*_L16.json |
| Adapter | configs/aper/ | aper_adapter_{cub,cifar,mini,inr}_*_L16.json |

### InfLoRA (CIL_Model/InfLoRA-main/configs/)

| 数据集 | 配置文件 |
|--------|----------|
| CUB-200 | `inflora_cub_B100_Inc10_L16.json` |
| CIFAR-100 | `inflora_cifar_B60_Inc5_L16.json` |
| miniImageNet | `inflora_mini_B60_Inc5_L16.json` |
| ImageNet-R | `inflora_inr_B100_Inc10_L16.json` |

### SD-LoRA (CIL_Model/SD-Lora-CL-main/configs/)

| 数据集 | 配置文件 |
|--------|----------|
| CUB-200 | `sdlora_cub_B100_Inc10_L16.json` |
| CIFAR-100 | `sdlora_cifar_B60_Inc5_L16.json` |
| miniImageNet | `sdlora_mini_B60_Inc5_L16.json` |
| ImageNet-R | `sdlora_inr_B100_Inc10_L16.json` |

### FSCIL-ASP & SEC-Prompt

| 项目 | 配置文件示例 |
|------|-------------|
| ASP | `asp_{cub,cifar,mini,inr}_*_L16.json` |
| SEC-Prompt | `SEC-Prompt_{cub,cifar,mini,inr}_*_L16.json` |

---

## 运行命令

```bash
# L2P
cd CIL_Model && python main.py --config configs/l2p/l2p_cub_B100_Inc10_L16.json

# DualPrompt
cd CIL_Model && python main.py --config configs/dualprompt/dualprompt_cub_B100_Inc10_L16.json

# Coda_Prompt
cd CIL_Model && python main.py --config configs/coda_prompt/coda_prompt_cub_B100_Inc10_L16.json

# EASE
cd CIL_Model && python main.py --config configs/ease/ease_cub_B100_Inc10_L16.json

# Adapter
cd CIL_Model && python main.py --config configs/aper/aper_adapter_cub_B100_Inc10_L16.json

# InfLoRA
cd CIL_Model/InfLoRA-main && python main.py --config configs/inflora_cub_B100_Inc10_L16.json

# SD-LoRA
cd CIL_Model/SD-Lora-CL-main && python main.py --config=./configs/sdlora_cub_B100_Inc10_L16.json

# ASP
cd FSCIL-ASP-main && python main.py --config configs/asp_cub_B100_Inc10_L16.json

# SEC-Prompt
cd SEC-Prompt-main && python main.py --config configs/SEC-Prompt_cub_B100_Inc10_L16.json
```

---

## InfLoRA 特殊说明

InfLoRA 通过 config 中的 `embd_dim` 参数自动选择模型：

```json
{
  "embd_dim": 1024,    // 触发 ViT-Large
  "num_heads": 16,     // 必须与 embd_dim 匹配
  ...
}
```

权重通过 timm 自动从 GitHub 下载，无需手动设置 backbone_type。
