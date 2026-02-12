"""
Fix configuration files:
1. Properly format arrays inline (device, seed, init_milestones, milestones)
2. Ensure dataset name matches base config exactly
3. Standardize device parameter within each method
"""
import json
import os
import re
from pathlib import Path

BASE_DIR = Path(r"f:\Repositories\Research\Awesome-FSCIL")

def format_json_inline(data):
    """Convert JSON to string with arrays formatted inline."""
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    
    # Pattern to match multi-line arrays with simple values
    # This regex matches arrays that span multiple lines with simple elements
    def inline_array(match):
        content = match.group(0)
        # Extract array content and format inline
        # Find the key and the array
        parts = content.split('[')
        if len(parts) >= 2:
            key_part = parts[0]
            array_content = content[len(key_part)+1:-1]  # content between [ and ]
            # Clean up the array content
            items = []
            for item in array_content.split(','):
                item = item.strip()
                if item:
                    items.append(item)
            if items:
                # Rebuild the line 
                return key_part + '[' + ', '.join(items) + ']'
        return content
    
    # Replace multi-line arrays with inline format
    lines = json_str.split('\n')
    result_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Check if this line starts an array
        if '": [' in line and not line.rstrip().endswith('],') and not line.rstrip().endswith(']'):
            # Collect all lines until the array closes
            array_lines = [line]
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(']'):
                array_lines.append(lines[i])
                i += 1
            if i < len(lines):
                array_lines.append(lines[i])
            
            # Parse the key and values
            combined = ''.join(array_lines)
            # Extract key
            key_match = re.match(r'(\s*"[^"]+": )\[', array_lines[0])
            if key_match:
                key_part = key_match.group(1)
                # Extract values
                values = []
                for al in array_lines[1:-1]:
                    val = al.strip().rstrip(',')
                    if val:
                        values.append(val)
                # Check if last line before ] has a value
                last_line = array_lines[-1].strip()
                if last_line.startswith(']'):
                    closing = last_line
                else:
                    closing = ']' + (',' if last_line.endswith(',') else '')
                
                inline_str = key_part + '[' + ', '.join(values) + ']'
                if array_lines[-1].strip().endswith(','):
                    inline_str += ','
                result_lines.append(inline_str)
            else:
                result_lines.extend(array_lines)
        else:
            result_lines.append(line)
        i += 1
    
    return '\n'.join(result_lines)


def regenerate_config(base_config_path, target_suffix, backbone_changes, other_changes=None):
    """Regenerate a config file based on base config with proper formatting."""
    if not base_config_path.exists():
        print(f"  Base not found: {base_config_path.name}")
        return False
    
    # Read original file as text to preserve formatting
    with open(base_config_path, 'r', encoding='utf-8') as f:
        base_text = f.read()
    
    # Parse JSON to get data
    base_data = json.loads(base_text)
    
    # Apply changes
    new_data = base_data.copy()
    new_data.update(backbone_changes)
    if other_changes:
        new_data.update(other_changes)
    
    # Generate target path
    target_name = base_config_path.stem + target_suffix + '.json'
    target_path = base_config_path.parent / target_name
    
    # Format JSON with inline arrays
    formatted = format_json_inline(new_data)
    
    with open(target_path, 'w', encoding='utf-8') as f:
        f.write(formatted)
    
    print(f"  Fixed: {target_name}")
    return True


def fix_all_configs():
    """Fix all generated configuration files."""
    print("=" * 60)
    print("Fixing Configuration Files")
    print("=" * 60)
    
    fixed_count = 0
    
    # SD-LoRA configs
    print("\n[SD-LoRA]")
    sdlora_dir = BASE_DIR / "CIL_Model" / "SD-Lora-CL-main" / "configs"
    sdlora_datasets = ["cub_B100_Inc10", "cifar_B60_Inc5", "mini_B60_Inc5", "inr_B100_Inc10"]
    
    for ds in sdlora_datasets:
        base_path = sdlora_dir / f"sdlora_{ds}.json"
        if not base_path.exists():
            continue
        
        with open(base_path, 'r', encoding='utf-8') as f:
            base_data = json.load(f)
        
        # Get device from base
        base_device = base_data.get("device", ["1"])
        
        # DINO config
        dino_path = sdlora_dir / f"sdlora_{ds}_dino.json"
        if dino_path.exists():
            dino_data = base_data.copy()
            dino_data["prefix"] = "dino"
            dino_data["backbone_type"] = "vit_base_patch16_224_dino"
            dino_data["device"] = base_device
            formatted = format_json_inline(dino_data)
            with open(dino_path, 'w', encoding='utf-8') as f:
                f.write(formatted)
            print(f"  Fixed: sdlora_{ds}_dino.json")
            fixed_count += 1
    
    # InfLoRA configs
    print("\n[InfLoRA]")
    inflora_dir = BASE_DIR / "CIL_Model" / "InfLoRA-main" / "configs"
    inflora_datasets = ["cub_B100_Inc10", "cifar_B60_Inc5", "mini_B60_Inc5", "inr_B100_Inc10"]
    
    for ds in inflora_datasets:
        base_path = inflora_dir / f"inflora_{ds}.json"
        if not base_path.exists():
            continue
        
        with open(base_path, 'r', encoding='utf-8') as f:
            base_data = json.load(f)
        
        base_device = base_data.get("device", ["1"])
        
        # DINO config
        dino_path = inflora_dir / f"inflora_{ds}_dino.json"
        if dino_path.exists():
            dino_data = base_data.copy()
            dino_data["prefix"] = "dino"
            dino_data["backbone_type"] = "vit_base_patch16_224_dino"
            dino_data["device"] = base_device
            formatted = format_json_inline(dino_data)
            with open(dino_path, 'w', encoding='utf-8') as f:
                f.write(formatted)
            print(f"  Fixed: inflora_{ds}_dino.json")
            fixed_count += 1
    
    # ASP configs
    print("\n[ASP]")
    asp_dir = BASE_DIR / "FSCIL-ASP-main" / "configs"
    asp_datasets = ["cub_B100_Inc10", "cifar_B60_Inc5", "mini_B60_Inc5", "inr_B100_Inc10"]
    
    for ds in asp_datasets:
        base_path = asp_dir / f"asp_{ds}.json"
        if not base_path.exists():
            continue
        
        with open(base_path, 'r', encoding='utf-8') as f:
            base_data = json.load(f)
        
        base_device = base_data.get("device", ["0"])
        
        # DINO config
        dino_path = asp_dir / f"asp_{ds}_dino.json"
        if dino_path.exists():
            dino_data = base_data.copy()
            dino_data["backbone_type"] = "pretrained_vit_b16_224_dino_vpt"
            dino_data["base_model_path"] = "asp_dino"
            dino_data["prefix"] = "asp_dino"
            dino_data["model_prefix"] = "asp_dino"
            dino_data["device"] = base_device
            formatted = format_json_inline(dino_data)
            with open(dino_path, 'w', encoding='utf-8') as f:
                f.write(formatted)
            print(f"  Fixed: asp_{ds}_dino.json")
            fixed_count += 1
    
    # SEC-Prompt configs
    print("\n[SEC-Prompt]")
    sec_dir = BASE_DIR / "SEC-Prompt-main" / "configs"
    sec_datasets = ["cub_B100_Inc10", "cifar_B60_Inc5", "mini_B60_Inc5", "inr_B100_Inc10"]
    
    for ds in sec_datasets:
        base_path = sec_dir / f"SEC-Prompt_{ds}.json"
        if not base_path.exists():
            continue
        
        with open(base_path, 'r', encoding='utf-8') as f:
            base_data = json.load(f)
        
        base_device = base_data.get("device", ["3"])
        
        # DINO config
        dino_path = sec_dir / f"SEC-Prompt_{ds}_dino.json"
        if dino_path.exists():
            dino_data = base_data.copy()
            dino_data["backbone_type"] = "pretrained_vit_b16_224_dino_vpt"
            dino_data["base_model_path"] = "sec_dino"
            dino_data["prefix"] = "sec_dino"
            dino_data["model_prefix"] = "sec_dino"
            dino_data["device"] = base_device
            formatted = format_json_inline(dino_data)
            with open(dino_path, 'w', encoding='utf-8') as f:
                f.write(formatted)
            print(f"  Fixed: SEC-Prompt_{ds}_dino.json")
            fixed_count += 1
    
    # L2P, DualPrompt, Coda_Prompt, EASE configs
    for method, prefix in [("l2p", "l2p"), ("dualprompt", "dualprompt"), 
                           ("coda_prompt", "coda_prompt"), ("ease", "ease")]:
        print(f"\n[{method.upper()}]")
        method_dir = BASE_DIR / "CIL_Model" / "configs" / method
        
        datasets = {
            "cub_B100_Inc10": "cub",
            "cifar_B60_Inc5": "cifar",
            "mini_B60_Inc5": "mini_imagenet",
            "inr_B100_Inc10": "imagenetr"
        }
        
        for ds, ds_name in datasets.items():
            base_path = method_dir / f"{prefix}_{ds}.json"
            if not base_path.exists():
                continue
            
            with open(base_path, 'r', encoding='utf-8') as f:
                base_data = json.load(f)
            
            base_device = base_data.get("device", ["3"])
            
            # DINO config
            dino_path = method_dir / f"{prefix}_{ds}_dino.json"
            if dino_path.exists():
                dino_data = base_data.copy()
                dino_data["prefix"] = "dino"
                if method == "ease":
                    dino_data["backbone_type"] = "vit_base_patch16_224_dino_ease"
                elif method == "l2p":
                    dino_data["backbone_type"] = "vit_base_patch16_224_dino_l2p"
                elif method == "dualprompt":
                    dino_data["backbone_type"] = "vit_base_patch16_224_dino_dualprompt"
                elif method == "coda_prompt":
                    dino_data["backbone_type"] = "vit_base_patch16_224_dino_coda_prompt"
                dino_data["device"] = base_device
                
                formatted = format_json_inline(dino_data)
                with open(dino_path, 'w', encoding='utf-8') as f:
                    f.write(formatted)
                print(f"  Fixed: {prefix}_{ds}_dino.json")
                fixed_count += 1
    
    # Fix EASE base configs
    print("\n[EASE Base Configs]")
    ease_dir = BASE_DIR / "CIL_Model" / "configs" / "ease"
    ease_datasets = {
        "cub_B100_Inc10": "cub",
        "cifar_B60_Inc5": "cifar",
        "mini_B60_Inc5": "mini_imagenet", 
        "inr_B100_Inc10": "imagenetr"
    }
    
    for ds, ds_name in ease_datasets.items():
        # Get reference from L16 config
        l16_path = ease_dir / f"ease_{ds}_L16.json"
        if not l16_path.exists():
            continue
        
        with open(l16_path, 'r', encoding='utf-8') as f:
            l16_data = json.load(f)
        
        base_device = l16_data.get("device", ["1"])
        
        # Fix base config
        base_path = ease_dir / f"ease_{ds}.json"
        if base_path.exists():
            base_data = l16_data.copy()
            base_data["backbone_type"] = "vit_base_patch16_224_ease"
            base_data["batch_size"] = l16_data.get("batch_size", 24) * 2
            base_data["prefix"] = l16_data.get("prefix", "exp").replace("_L16", "").replace("L16", "").strip() or "exp"
            base_data["device"] = base_device
            
            formatted = format_json_inline(base_data)
            with open(base_path, 'w', encoding='utf-8') as f:
                f.write(formatted)
            print(f"  Fixed: ease_{ds}.json")
            fixed_count += 1
        
        # Fix in21k config
        in21k_path = ease_dir / f"ease_{ds}_in21k.json"
        if in21k_path.exists():
            in21k_data = l16_data.copy()
            in21k_data["backbone_type"] = "vit_base_patch16_224_in21k_ease"
            in21k_data["batch_size"] = l16_data.get("batch_size", 24) * 2
            in21k_data["prefix"] = "in21k"
            in21k_data["device"] = base_device
            
            formatted = format_json_inline(in21k_data)
            with open(in21k_path, 'w', encoding='utf-8') as f:
                f.write(formatted)
            print(f"  Fixed: ease_{ds}_in21k.json")
            fixed_count += 1
    
    # Fix LAE L16 configs
    print("\n[LAE L16 Configs]")
    lae_dir = BASE_DIR / "CIL_Model" / "configs" / "lae"
    lae_datasets = ["cub_B100_Inc10", "cifar_B60_Inc5", "mini_B60_Inc5", "inr_B100_Inc10"]
    
    for ds in lae_datasets:
        base_path = lae_dir / f"lae_{ds}.json"
        l16_path = lae_dir / f"lae_{ds}_L16.json"
        
        if not base_path.exists() or not l16_path.exists():
            continue
        
        with open(base_path, 'r', encoding='utf-8') as f:
            base_data = json.load(f)
        
        base_device = base_data.get("device", ["3"])
        
        l16_data = base_data.copy()
        l16_data["backbone_type"] = base_data.get("backbone_type", "vit_base_patch16_224_lae").replace(
            "vit_base_patch16_224_", "vit_large_patch16_224_"
        )
        l16_data["batch_size"] = max(1, base_data.get("batch_size", 48) // 2)
        l16_data["device"] = base_device
        prefix = base_data.get("prefix", "").strip()
        l16_data["prefix"] = (prefix + "_L16") if prefix else "L16"
        
        formatted = format_json_inline(l16_data)
        with open(l16_path, 'w', encoding='utf-8') as f:
            f.write(formatted)
        print(f"  Fixed: lae_{ds}_L16.json")
        fixed_count += 1
    
    print("\n" + "=" * 60)
    print(f"Total configs fixed: {fixed_count}")
    print("=" * 60)


if __name__ == "__main__":
    fix_all_configs()
