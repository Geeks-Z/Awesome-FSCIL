# DINO ViT-B/16 模型支持方案

## 概述

本文档总结了为9个FSCIL/CIL方法添加DINO ViT-B/16预训练模型支持的修改方案。

**DINO模型信息：**
- **timm模型名**: `vit_base_patch16_224_dino`
- **输出维度**: 768 (与ImageNet-1K/21K的ViT-B/16相同)
- **架构**: 标准ViT-B/16 (patch=16, embed_dim=768, depth=12, num_heads=12)
- **预训练**: 自监督学习（无需标签）

---

## 方法支持状态

| 方法 | 代码位置 | DINO支持 | backbone_type |
|------|----------|----------|---------------|
| Full Finetune | CIL_Model | ✅ 已支持 | `vit_base_patch16_224_dino` |
| SimpleFSCIL | CIL_Model | ✅ 已支持 | `vit_base_patch16_224_dino` |
| L2P | CIL_Model | ✅ 已支持 | `vit_base_patch16_224_dino_l2p` |
| CODA-Prompt | CIL_Model | ✅ 已支持 | `vit_base_patch16_224_dino_coda_prompt` |
| DualPrompt | CIL_Model | ✅ 已支持 | `vit_base_patch16_224_dino_dualprompt` |
| LAE | CIL_Model | ✅ 已支持 | `vit_base_patch16_224_dino_lae` |
| InfLoRA | InfLoRA-main | ✅ 已添加 | `vit_base_patch16_224_dino` |
| SD-LoRA | SD-Lora-CL-main | ✅ 已添加 | `vit_base_patch16_224_dino` |
| ASP | FSCIL-ASP-main | ✅ 已添加 | `pretrained_vit_b16_224_dino_vpt` |
| SEC-Prompt | SEC-Prompt-main | ✅ 已添加 | `pretrained_vit_b16_224_dino_vpt` |

---

## 修改的文件

### 1. CIL_Model

**已修改文件：**
- `CIL_Model/backbone/vpt.py` - 添加DINO支持
- `CIL_Model/utils/inc_net.py` - 添加VPT的DINO分支

**已创建配置文件：**
- `configs/finetune/finetune_cub_B100_Inc10_dino.json`
- `configs/l2p/l2p_cub_B100_Inc10_dino.json`
- `configs/coda_prompt/coda_prompt_cub_B100_Inc10_dino.json`
- `configs/dualprompt/dualprompt_cub_B100_Inc10_dino.json`
- `configs/lae/lae_cub_B100_Inc10_dino.json`

**注：** SimpleCIL的DINO配置已存在于 `configs/simplecil/dino/` 目录

### 2. InfLoRA-main

**已修改文件：**
- `InfLoRA-main/utils/inc_net.py` - 添加DINO backbone加载

**已创建配置文件：**
- `configs/inflora_cub_B100_Inc10_dino.json`

### 3. SD-Lora-CL-main

**已修改文件：**
- `SD-Lora-CL-main/utils/inc_net.py` - 添加DINO backbone加载

**已创建配置文件：**
- `exps/sdlora_cub_B100_Inc10_dino.json`

### 4. FSCIL-ASP-main

**已修改文件：**
- `backbone/asp_backbone.py` - 添加DINO支持
- `utils/inc_net.py` - 添加DINO backbone和VPT分支

**已创建配置文件：**
- `configs/asp_cub_B100_Inc10_dino.json`

### 5. SEC-Prompt-main

**已修改文件：**
- `backbone/vpt_backbone.py` - 添加DINO支持
- `utils/inc_net.py` - 添加DINO backbone和VPT分支

**已创建配置文件：**
- `configs/SEC-Prompt_cub_B100_Inc10_dino.json`

---

## 使用方法

### CIL_Model 方法

```bash
# Full Finetune
python main.py --config configs/finetune/finetune_cub_B100_Inc10_dino.json

# SimpleFSCIL
python main.py --config configs/simplecil/dino/simplecil_cub_B100_Inc10.json

# L2P
python main.py --config configs/l2p/l2p_cub_B100_Inc10_dino.json

# CODA-Prompt
python main.py --config configs/coda_prompt/coda_prompt_cub_B100_Inc10_dino.json

# DualPrompt
python main.py --config configs/dualprompt/dualprompt_cub_B100_Inc10_dino.json

# LAE
python main.py --config configs/lae/lae_cub_B100_Inc10_dino.json
```

### InfLoRA

```bash
cd InfLoRA-main
python main.py --config configs/inflora_cub_B100_Inc10_dino.json
```

### SD-LoRA

```bash
cd SD-Lora-CL-main
python main.py --config configs/sdlora_cub_B100_Inc10_dino.json
```

### ASP

```bash
cd FSCIL-ASP-main
python main.py --config configs/asp_cub_B100_Inc10_dino.json
```

### SEC-Prompt

```bash
cd SEC-Prompt-main
python main.py --config configs/SEC-Prompt_cub_B100_Inc10_dino.json
```

---

## 创建其他数据集的DINO配置

以CUB配置为模板，只需修改以下字段：

### CIFAR-100 (B60 Inc5)
```json
{
  "dataset": "cifar",
  "init_cls": 60,
  "increment": 5
}
```

### miniImageNet (B60 Inc5)
```json
{
  "dataset": "mini",
  "init_cls": 60,
  "increment": 5
}
```

### ImageNet-R (B100 Inc10)
```json
{
  "dataset": "inr",
  "init_cls": 100,
  "increment": 10
}
```

---

## 技术说明

### DINO vs ImageNet预训练

1. **预训练方式**：
   - ImageNet-1K: 有监督分类（1000类）
   - ImageNet-21K: 有监督分类（21843类）
   - DINO: 自监督学习（无需标签）

2. **模型结构**：
   - 三者使用相同的ViT-B/16架构
   - 输出维度均为768

3. **特性**：
   - DINO模型没有预训练的分类头（num_classes=0）
   - DINO特征具有良好的语义聚类特性
   - 适合few-shot和迁移学习任务

### 注意事项

1. **timm版本兼容性**：
   - timm 0.6+: 使用 `vit_base_patch16_224_dino`
   - timm 0.9+: 也可使用 `vit_base_patch16_224.dino`

2. **首次运行**：
   - 模型权重会自动从Facebook下载
   - 确保网络连接正常

3. **显存需求**：
   - DINO模型与ImageNet预训练模型显存需求相同
