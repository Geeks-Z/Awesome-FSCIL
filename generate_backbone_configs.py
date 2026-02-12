"""
Batch generate missing configuration files for all methods and backbones.
This script generates DINO, In21K, and ViT-Large (L16) configurations 
for each dataset based on existing base configurations.
"""
import json
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(r"f:\Repositories\Research\Awesome-FSCIL")

# Backbone configurations for CIL_Model methods (L2P, DualPrompt, Coda_Prompt, EASE, LAE)
CIL_BACKBONE_CONFIGS = {
    "base": {
        "suffix": "",
        "prefix_update": None,  # Keep original
        "batch_size_factor": 1.0,
    },
    "dino": {
        "suffix": "_dino",
        "prefix_update": "dino",
        "batch_size_factor": 1.0,
        "backbone_modifier": lambda s: s.replace("_base_patch16_224_", "_base_patch16_224_dino_"),
    },
    "in21k": {
        "suffix": "_in21k",
        "prefix_update": None,  # Optional
        "batch_size_factor": 1.0,
        "backbone_modifier": lambda s: s.replace("_base_patch16_224_", "_base_patch16_224_in21k_"),
    },
    "L16": {
        "suffix": "_L16",
        "prefix_update": lambda p: p.strip() + "_L16" if p.strip() else "L16",
        "batch_size_factor": 0.5,
        "backbone_modifier": lambda s: s.replace("vit_base_patch16_224_", "vit_large_patch16_224_"),
    },
}

# Dataset settings for config generation
DATASETS = {
    "cub": ("cub", "B100_Inc10"),
    "cifar": ("cifar", "B60_Inc5"),
    "mini": ("mini_imagenet", "B60_Inc5"),
    "inr": ("imagenetr", "B100_Inc10"),
}

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Created: {path.name}")

def generate_cil_model_configs(method_dir, method_prefix, backbone_types):
    """Generate configs for CIL_Model methods (L2P, DualPrompt, etc.)"""
    config_dir = BASE_DIR / "CIL_Model" / "configs" / method_dir
    
    if not config_dir.exists():
        print(f"Directory not found: {config_dir}")
        return
    
    generated = 0
    for ds_short, (ds_name, setting) in DATASETS.items():
        base_config_name = f"{method_prefix}_{ds_short}_{setting}.json"
        base_config_path = config_dir / base_config_name
        
        if not base_config_path.exists():
            # Try to find alternative base
            alt_formats = [
                f"{method_prefix}_{ds_short}_{setting.split('_')[0]}_Inc5.json",
                f"{method_prefix}_{ds_short}_{setting.split('_')[0]}_Inc10.json",
            ]
            for alt in alt_formats:
                if (config_dir / alt).exists():
                    base_config_path = config_dir / alt
                    base_config_name = alt
                    break
            else:
                print(f"  Base config not found for {ds_short}: tried {base_config_name}")
                continue
        
        base_config = load_json(base_config_path)
        
        for variant, cfg in backbone_types.items():
            if variant == "base":
                continue
                
            new_config_name = base_config_name.replace(".json", f"{cfg['suffix']}.json")
            new_config_path = config_dir / new_config_name
            
            if new_config_path.exists():
                continue  # Skip existing
            
            new_config = base_config.copy()
            
            # Update backbone_type
            if "backbone_modifier" in cfg and "backbone_type" in new_config:
                new_config["backbone_type"] = cfg["backbone_modifier"](new_config["backbone_type"])
            
            # Update prefix
            if cfg["prefix_update"] is not None:
                if callable(cfg["prefix_update"]):
                    new_config["prefix"] = cfg["prefix_update"](new_config.get("prefix", ""))
                else:
                    new_config["prefix"] = cfg["prefix_update"]
            
            # Update batch_size
            if cfg["batch_size_factor"] != 1.0 and "batch_size" in new_config:
                new_config["batch_size"] = int(new_config["batch_size"] * cfg["batch_size_factor"])
            
            save_json(new_config, new_config_path)
            generated += 1
    
    return generated


def generate_ease_configs():
    """Generate EASE configs - only L16 exists, need base/dino/in21k."""
    config_dir = BASE_DIR / "CIL_Model" / "configs" / "ease"
    
    if not config_dir.exists():
        print("EASE config directory not found")
        return 0
    
    generated = 0
    
    # First, generate base configs from L16
    for ds_short, (ds_name, setting) in DATASETS.items():
        l16_config_name = f"ease_{ds_short}_{setting}_L16.json"
        l16_config_path = config_dir / l16_config_name
        
        if not l16_config_path.exists():
            print(f"  L16 config not found: {l16_config_name}")
            continue
        
        l16_config = load_json(l16_config_path)
        
        # Define target configs
        variants = [
            ("", "vit_base_patch16_224_ease", 2.0, None),
            ("_dino", "vit_base_patch16_224_dino_ease", 2.0, "dino"),
            ("_in21k", "vit_base_patch16_224_in21k_ease", 2.0, None),
        ]
        
        for suffix, backbone_type, batch_factor, prefix_update in variants:
            new_config_name = f"ease_{ds_short}_{setting}{suffix}.json"
            new_config_path = config_dir / new_config_name
            
            if new_config_path.exists():
                continue
            
            new_config = l16_config.copy()
            new_config["backbone_type"] = backbone_type
            
            if "batch_size" in new_config:
                new_config["batch_size"] = int(new_config["batch_size"] * batch_factor)
            
            if prefix_update is not None:
                new_config["prefix"] = prefix_update
            else:
                # Remove L16 from prefix if present
                if "prefix" in new_config:
                    new_config["prefix"] = new_config["prefix"].replace("_L16", "").replace("L16", "").strip()
                    if not new_config["prefix"]:
                        new_config["prefix"] = " "
            
            save_json(new_config, new_config_path)
            generated += 1
    
    return generated


def generate_lae_configs():
    """Generate LAE configs for dino and L16."""
    config_dir = BASE_DIR / "CIL_Model" / "configs" / "lae"
    
    if not config_dir.exists():
        print("LAE config directory not found")
        return 0
    
    generated = 0
    
    for ds_short, (ds_name, setting) in DATASETS.items():
        base_config_name = f"lae_{ds_short}_{setting}.json"
        base_config_path = config_dir / base_config_name
        
        if not base_config_path.exists():
            print(f"  Base config not found: {base_config_name}")
            continue
        
        base_config = load_json(base_config_path)
        
        # Generate L16 config
        l16_config_name = f"lae_{ds_short}_{setting}_L16.json"
        l16_config_path = config_dir / l16_config_name
        
        if not l16_config_path.exists():
            l16_config = base_config.copy()
            if "backbone_type" in l16_config:
                l16_config["backbone_type"] = l16_config["backbone_type"].replace(
                    "vit_base_patch16_224_", "vit_large_patch16_224_"
                )
            if "batch_size" in l16_config:
                l16_config["batch_size"] = int(l16_config["batch_size"] * 0.5)
            if "prefix" in l16_config:
                l16_config["prefix"] = l16_config["prefix"].strip() + "_L16" if l16_config["prefix"].strip() else "L16"
            
            save_json(l16_config, l16_config_path)
            generated += 1
    
    return generated


def generate_inflora_configs():
    """Generate InfLoRA configs for missing dino/in21k/L16."""
    config_dir = BASE_DIR / "CIL_Model" / "InfLoRA-main" / "configs"
    
    if not config_dir.exists():
        print("InfLoRA config directory not found")
        return 0
    
    generated = 0
    
    inflora_datasets = {
        "cub": ("cub", "B100_Inc10"),
        "cifar": ("cifar", "B60_Inc5"),
        "mini": ("mini", "B60_Inc5"),
        "inr": ("inr", "B100_Inc10"),
    }
    
    for ds_short, (ds_name, setting) in inflora_datasets.items():
        base_config_name = f"inflora_{ds_short}_{setting}.json"
        base_config_path = config_dir / base_config_name
        
        if not base_config_path.exists():
            print(f"  Base config not found: {base_config_name}")
            continue
        
        base_config = load_json(base_config_path)
        
        # Generate dino config
        dino_config_name = f"inflora_{ds_short}_{setting}_dino.json"
        if not (config_dir / dino_config_name).exists():
            dino_config = base_config.copy()
            dino_config["prefix"] = "dino"
            dino_config["backbone_type"] = "vit_base_patch16_224_dino"
            save_json(dino_config, config_dir / dino_config_name)
            generated += 1
        
        # Generate in21k config
        in21k_config_name = f"inflora_{ds_short}_{setting}_in21k.json"
        if not (config_dir / in21k_config_name).exists():
            in21k_config = base_config.copy()
            in21k_config["backbone_type"] = "vit_base_patch16_224_in21k"
            save_json(in21k_config, config_dir / in21k_config_name)
            generated += 1
        
        # Generate L16 config
        l16_config_name = f"inflora_{ds_short}_{setting}_L16.json"
        if not (config_dir / l16_config_name).exists():
            l16_config = base_config.copy()
            l16_config["prefix"] = l16_config.get("prefix", "reproduce") + "_L16"
            l16_config["embd_dim"] = 1024
            l16_config["num_heads"] = 16
            if "batch_size" in l16_config:
                l16_config["batch_size"] = int(l16_config["batch_size"] * 0.5)
            save_json(l16_config, config_dir / l16_config_name)
            generated += 1
    
    return generated


def generate_sdlora_configs():
    """Generate SD-LoRA configs for missing dino/in21k/L16."""
    config_dir = BASE_DIR / "CIL_Model" / "SD-Lora-CL-main" / "configs"
    
    if not config_dir.exists():
        print("SD-LoRA config directory not found")
        return 0
    
    generated = 0
    
    sdlora_datasets = {
        "cub": ("cub", "B100_Inc10"),
        "cifar": ("cifar", "B60_Inc5"),
        "mini": ("mini", "B60_Inc5"),
        "inr": ("inr", "B100_Inc10"),
    }
    
    for ds_short, (ds_name, setting) in sdlora_datasets.items():
        base_config_name = f"sdlora_{ds_short}_{setting}.json"
        base_config_path = config_dir / base_config_name
        
        if not base_config_path.exists():
            print(f"  Base config not found: {base_config_name}")
            continue
        
        base_config = load_json(base_config_path)
        
        # Generate dino config
        dino_config_name = f"sdlora_{ds_short}_{setting}_dino.json"
        if not (config_dir / dino_config_name).exists():
            dino_config = base_config.copy()
            dino_config["prefix"] = "dino"
            dino_config["backbone_type"] = "vit_base_patch16_224_dino"
            save_json(dino_config, config_dir / dino_config_name)
            generated += 1
        
        # Generate in21k config
        in21k_config_name = f"sdlora_{ds_short}_{setting}_in21k.json"
        if not (config_dir / in21k_config_name).exists():
            in21k_config = base_config.copy()
            in21k_config["backbone_type"] = "vit_base_patch16_224_in21k"
            save_json(in21k_config, config_dir / in21k_config_name)
            generated += 1
        
        # Generate L16 config
        l16_config_name = f"sdlora_{ds_short}_{setting}_L16.json"
        if not (config_dir / l16_config_name).exists():
            l16_config = base_config.copy()
            l16_config["prefix"] = l16_config.get("prefix", "reproduce") + "_L16"
            l16_config["backbone_type"] = "vit_large_patch16_224"
            if "batch_size" in l16_config:
                l16_config["batch_size"] = int(l16_config["batch_size"] * 0.5)
            save_json(l16_config, config_dir / l16_config_name)
            generated += 1
    
    return generated


def generate_asp_configs():
    """Generate ASP configs for missing dino/in21k/L16."""
    config_dir = BASE_DIR / "FSCIL-ASP-main" / "configs"
    
    if not config_dir.exists():
        print("ASP config directory not found")
        return 0
    
    generated = 0
    
    asp_datasets = {
        "cub": ("cub", "B100_Inc10"),
        "cifar": ("cifar", "B60_Inc5"),
        "mini": ("mini", "B60_Inc5"),
        "inr": ("inr", "B100_Inc10"),
    }
    
    for ds_short, (ds_name, setting) in asp_datasets.items():
        base_config_name = f"asp_{ds_short}_{setting}.json"
        base_config_path = config_dir / base_config_name
        
        if not base_config_path.exists():
            print(f"  Base config not found: {base_config_name}")
            continue
        
        base_config = load_json(base_config_path)
        
        # Generate dino config
        dino_config_name = f"asp_{ds_short}_{setting}_dino.json"
        if not (config_dir / dino_config_name).exists():
            dino_config = base_config.copy()
            dino_config["backbone_type"] = "pretrained_vit_b16_224_dino_vpt"
            dino_config["base_model_path"] = "asp_dino"
            dino_config["prefix"] = "asp_dino"
            dino_config["model_prefix"] = "asp_dino"
            save_json(dino_config, config_dir / dino_config_name)
            generated += 1
        
        # Generate in21k config
        in21k_config_name = f"asp_{ds_short}_{setting}_in21k.json"
        if not (config_dir / in21k_config_name).exists():
            in21k_config = base_config.copy()
            in21k_config["backbone_type"] = "pretrained_vit_b16_224_in21k_vpt"
            in21k_config["base_model_path"] = "asp_in21k"
            in21k_config["prefix"] = "asp_in21k"
            in21k_config["model_prefix"] = "asp_in21k"
            save_json(in21k_config, config_dir / in21k_config_name)
            generated += 1
        
        # Generate L16 config
        l16_config_name = f"asp_{ds_short}_{setting}_L16.json"
        if not (config_dir / l16_config_name).exists():
            l16_config = base_config.copy()
            l16_config["backbone_type"] = "pretrained_vit_large_patch16_224_vpt"
            l16_config["base_model_path"] = "asp_L16"
            l16_config["prefix"] = "asp_L16"
            l16_config["model_prefix"] = "asp_L16"
            if "batch_size" in l16_config:
                l16_config["batch_size"] = int(l16_config["batch_size"] * 0.67)  # 24 -> 16
            if "fs_batch_size" in l16_config:
                l16_config["fs_batch_size"] = int(l16_config["fs_batch_size"] * 0.5)
            save_json(l16_config, config_dir / l16_config_name)
            generated += 1
    
    return generated


def generate_sec_prompt_configs():
    """Generate SEC-Prompt configs for missing dino/in21k/L16."""
    config_dir = BASE_DIR / "SEC-Prompt-main" / "configs"
    
    if not config_dir.exists():
        print("SEC-Prompt config directory not found")
        return 0
    
    generated = 0
    
    sec_datasets = {
        "cub": ("cub", "B100_Inc10"),
        "cifar": ("cifar", "B60_Inc5"),
        "mini": ("mini", "B60_Inc5"),
        "inr": ("inr", "B100_Inc10"),
    }
    
    for ds_short, (ds_name, setting) in sec_datasets.items():
        base_config_name = f"SEC-Prompt_{ds_short}_{setting}.json"
        base_config_path = config_dir / base_config_name
        
        if not base_config_path.exists():
            print(f"  Base config not found: {base_config_name}")
            continue
        
        base_config = load_json(base_config_path)
        
        # Generate dino config
        dino_config_name = f"SEC-Prompt_{ds_short}_{setting}_dino.json"
        if not (config_dir / dino_config_name).exists():
            dino_config = base_config.copy()
            dino_config["backbone_type"] = "pretrained_vit_b16_224_dino_vpt"
            dino_config["base_model_path"] = "sec_dino"
            dino_config["prefix"] = "sec_dino"
            dino_config["model_prefix"] = "sec_dino"
            save_json(dino_config, config_dir / dino_config_name)
            generated += 1
        
        # Generate in21k config
        in21k_config_name = f"SEC-Prompt_{ds_short}_{setting}_in21k.json"
        if not (config_dir / in21k_config_name).exists():
            in21k_config = base_config.copy()
            in21k_config["backbone_type"] = "pretrained_vit_b16_224_in21k_vpt"
            in21k_config["base_model_path"] = "sec_in21k"
            in21k_config["prefix"] = "sec_in21k"
            in21k_config["model_prefix"] = "sec_in21k"
            save_json(in21k_config, config_dir / in21k_config_name)
            generated += 1
        
        # Generate L16 config
        l16_config_name = f"SEC-Prompt_{ds_short}_{setting}_L16.json"
        if not (config_dir / l16_config_name).exists():
            l16_config = base_config.copy()
            l16_config["backbone_type"] = "pretrained_vit_large_patch16_224_vpt"
            l16_config["base_model_path"] = "sec_L16"
            l16_config["prefix"] = "sec_L16" 
            l16_config["model_prefix"] = "sec_L16"
            if "batch_size" in l16_config:
                l16_config["batch_size"] = int(l16_config["batch_size"] * 0.5)
            if "fs_batch_size" in l16_config:
                l16_config["fs_batch_size"] = int(l16_config["fs_batch_size"] * 0.5)
            save_json(l16_config, config_dir / l16_config_name)
            generated += 1
    
    return generated


def main():
    print("=" * 60)
    print("Generating Multi-Backbone Configuration Files")
    print("=" * 60)
    
    total = 0
    
    # CIL_Model methods
    print("\n[L2P]")
    total += generate_cil_model_configs("l2p", "l2p", CIL_BACKBONE_CONFIGS)
    
    print("\n[DualPrompt]")
    total += generate_cil_model_configs("dualprompt", "dualprompt", CIL_BACKBONE_CONFIGS)
    
    print("\n[Coda_Prompt]")
    total += generate_cil_model_configs("coda_prompt", "coda_prompt", CIL_BACKBONE_CONFIGS)
    
    print("\n[EASE]")
    total += generate_ease_configs()
    
    print("\n[LAE]")
    total += generate_lae_configs()
    
    # Standalone methods
    print("\n[InfLoRA]")
    total += generate_inflora_configs()
    
    print("\n[SD-LoRA]")
    total += generate_sdlora_configs()
    
    print("\n[ASP]")
    total += generate_asp_configs()
    
    print("\n[SEC-Prompt]")
    total += generate_sec_prompt_configs()
    
    print("\n" + "=" * 60)
    print(f"Total new configuration files generated: {total}")
    print("=" * 60)


if __name__ == "__main__":
    main()
