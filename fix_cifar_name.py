"""Fix CIFAR dataset names from 'cifar' to 'cifar224'"""
import os

files_to_fix = [
    r'CIL_Model\InfLoRA-main\configs\inflora_cifar_B60_Inc5_L16.json',
    r'CIL_Model\SD-Lora-CL-main\configs\sdlora_cifar_B60_Inc5_L16.json',
    r'FSCIL-ASP-main\configs\asp_cifar_B60_Inc5_L16.json',
    r'SEC-Prompt-main\configs\SEC-Prompt_cifar_B60_Inc5_L16.json',
    r'CIL_Model\configs\aper\aper_adapter_cifar_B60_Inc5_L16.json',
]

for f in files_to_fix:
    if os.path.exists(f):
        with open(f, 'r', encoding='utf-8') as fp:
            content = fp.read()
        content = content.replace('"dataset": "cifar"', '"dataset": "cifar224"')
        with open(f, 'w', encoding='utf-8') as fp:
            fp.write(content)
        print(f'Fixed: {os.path.basename(f)}')
    else:
        print(f'Not found: {f}')
