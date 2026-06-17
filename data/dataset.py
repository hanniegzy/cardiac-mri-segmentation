"""
M&Ms数据集加载器
支持心脏MRI图像和标注的加载和处理
"""

import os
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import nibabel as nib
from torch.utils.data import Dataset
import torch


class MNMSDataset(Dataset):
    """
    M&Ms (Multi-Centre, Multi-Vendor & Multi-Disease) 数据集加载器
    
    目标分割类别:
    - 0: 背景
    - 1: 左心室 (LV - Left Ventricle)
    - 2: 右心室 (RV - Right Ventricle)
    - 3: 心肌 (MYO - Myocardium)
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        augmentation=None,
        preprocessor=None,
        image_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        normalize_mean: float = 0.5,
        normalize_std: float = 0.5,
    ):
        """
        初始化M&Ms数据集
        
        Args:
            data_dir: 数据集根目录
            split: 数据集分割 ('train', 'val', 'test')
            augmentation: 数据增强对象
            preprocessor: 预处理对象
            image_size: 输入图像尺寸 (height, width)
            normalize: 是否进行标准化
            normalize_mean: 标准化均值
            normalize_std: 标准化标准差
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.augmentation = augmentation
        self.preprocessor = preprocessor
        self.image_size = image_size
        self.normalize = normalize
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        
        # 加载数据集文件列表
        self.image_files, self.label_files = self._load_dataset()
        
        if len(self.image_files) == 0:
            raise RuntimeError(f"No data found in {data_dir} for split {split}")
    
    def _load_dataset(self) -> Tuple[List[str], List[str]]:
        """
        从数据目录加载数据集
        
        Returns:
            (image_files, label_files) 元组
        """
        image_files = []
        label_files = []
        
        # 根据数据集结构调整路径
        split_dir = self.data_dir / self.split
        
        if split_dir.exists():
            # 标准结构: data/split/images, data/split/labels
            image_dir = split_dir / "images"
            label_dir = split_dir / "labels"
            
            if image_dir.exists() and label_dir.exists():
                for img_file in sorted(image_dir.glob("*.nii.gz")):
                    base_name = img_file.stem.replace('.nii', '')
                    label_file = label_dir / f"{base_name}_gt.nii.gz"
                    
                    if label_file.exists():
                        image_files.append(str(img_file))
                        label_files.append(str(label_file))
        
        return image_files, label_files
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个数据样本
        
        Args:
            idx: 样本索引
            
        Returns:
            包含 'image' 和 'label' 的字典
        """
        # 加载NIfTI图像
        img_path = self.image_files[idx]
        label_path = self.label_files[idx]
        
        # 读取NIfTI文件
        img_nii = nib.load(img_path)
        label_nii = nib.load(label_path)
        
        image = img_nii.get_fdata().astype(np.float32)
        label = label_nii.get_fdata().astype(np.int64)
        
        # 如果是3D图像，取中间切片或随机切片
        if image.ndim == 3:
            if self.split == "train":
                # 训练时随机选择切片
                slice_idx = np.random.randint(0, image.shape[-1])
            else:
                # 验证/测试时选择中间切片
                slice_idx = image.shape[-1] // 2
            
            image = image[:, :, slice_idx]
            label = label[:, :, slice_idx]
        
        # 预处理
        if self.preprocessor is not None:
            image, label = self.preprocessor(image, label)
        else:
            # 默认预处理：resize和标准化
            image = self._resize_image(image, self.image_size)
            label = self._resize_label(label, self.image_size)
            
            if self.normalize:
                image = (image - self.normalize_mean) / self.normalize_std
        
        # 数据增强（仅在训练集）
        if self.split == "train" and self.augmentation is not None:
            image, label = self.augmentation(image, label)
        
        # 转换为张量
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()
        
        # 添加通道维度
        if image.ndim == 2:
            image = image.unsqueeze(0)
        
        return {
            "image": image,
            "label": label,
            "image_path": img_path,
            "label_path": label_path,
        }
    
    @staticmethod
    def _resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """调整图像大小（双线性插值）"""
        from cv2 import resize, INTER_LINEAR
        return resize(image, (target_size[1], target_size[0]), interpolation=INTER_LINEAR)
    
    @staticmethod
    def _resize_label(label: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """调整标注大小（最近邻插值保持整数值）"""
        from cv2 import resize, INTER_NEAREST
        return resize(label.astype(np.float32), (target_size[1], target_size[0]), 
                      interpolation=INTER_NEAREST).astype(np.int64)
    
    def get_class_weights(self) -> np.ndarray:
        """计算类别权重（用于处理类不平衡）"""
        class_counts = np.zeros(4)  # 4个类别: BG, LV, RV, MYO
        
        for label_file in self.label_files:
            label_nii = nib.load(label_file)
            label = label_nii.get_fdata().astype(np.int64)
            
            for class_id in range(4):
                class_counts[class_id] += np.sum(label == class_id)
        
        # 计算权重（反比例）
        total = class_counts.sum()
        weights = total / (class_counts * len(np.where(class_counts > 0)[0]))
        
        return weights / weights.sum()


def create_dataloaders(
    data_dir: str,
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (224, 224),
    augmentation_config: Optional[Dict] = None,
) -> Tuple:
    """
    创建训练、验证和测试数据加载器
    
    Args:
        data_dir: 数据集根目录
        batch_size: 批大小
        num_workers: 数据加载工作进程数
        image_size: 输入图像尺寸
        augmentation_config: 数据增强配置
        
    Returns:
        (train_loader, val_loader, test_loader) 元组
    """
    from torch.utils.data import DataLoader
    from .augmentation import CardiacAugmentation
    from .preprocessor import CardiacPreprocessor
    
    # 创建预处理器
    preprocessor = CardiacPreprocessor(image_size=image_size)
    
    # 创建训练集数据增强
    train_augmentation = None
    if augmentation_config and augmentation_config.get('train', {}).get('enable', False):
        train_augmentation = CardiacAugmentation(augmentation_config['train'])
    
    # 创建数据集
    train_dataset = MNMSDataset(
        data_dir=data_dir,
        split="train",
        augmentation=train_augmentation,
        preprocessor=preprocessor,
        image_size=image_size,
    )
    
    val_dataset = MNMSDataset(
        data_dir=data_dir,
        split="val",
        augmentation=None,
        preprocessor=preprocessor,
        image_size=image_size,
    )
    
    test_dataset = MNMSDataset(
        data_dir=data_dir,
        split="test",
        augmentation=None,
        preprocessor=preprocessor,
        image_size=image_size,
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader
