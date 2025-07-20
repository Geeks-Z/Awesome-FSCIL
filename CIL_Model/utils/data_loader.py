import numpy as np
from torch.utils.data import DataLoader
from typing import List, Optional


def get_data_loaders(
        data_manager,
        known_classes: int,
        total_classes: int,
        batch_size: int,
        kshot: Optional[int] = None,
        num_workers: int = 4,
        multiple_gpus: bool = False
) -> dict:
    """
    创建并返回训练、测试和 protonet 训练所需的 DataLoader

    参数:
        data_manager: 数据管理对象
        known_classes: 已知类别数
        total_classes: 总类别数
        batch_size: 批量大小
        kshot: few-shot 学习中的 k 值
        num_workers: DataLoader 的工作线程数
        multiple_gpus: 是否使用多GPU

    返回:
        包含三个 DataLoader 的字典:
        {
            "train": 训练集 DataLoader,
            "test": 测试集 DataLoader,
            "protonet": protonet 训练集 DataLoader
        }
    """
    # 训练集
    train_dataset = data_manager.get_dataset(
        np.arange(known_classes, total_classes),
        source="train", mode="train", kshot=kshot
    )

    # 测试集
    test_dataset = data_manager.get_dataset(
        np.arange(0, total_classes),
        source="test", mode="test"
    )

    # Protonet 训练集
    # train_dataset_for_protonet = data_manager.get_dataset(
    #     np.arange(known_classes, total_classes),
    #     source="train", mode="test"
    # )

    # 创建 DataLoader
    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=multiple_gpus  # 多GPU时启用pin_memory
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=multiple_gpus
        ),
        "protonet": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=multiple_gpus
        )
    }

    return loaders