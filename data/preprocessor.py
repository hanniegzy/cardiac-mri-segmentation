"""
数据预处理模块
包含图像归一化、裁剪、调整大小等操作
"""

import numpy as np
import cv2
from typing import Tuple, Optional


class CardiacPreprocessor:
    """心脏MRI图像预处理器"""
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        normalize_mean: float = 0.5,
        normalize_std: float = 0.5,
        clip_range: Optional[Tuple[float, float]] = None,
    ):
        """
        初始化预处理器
        
        Args:
            image_size: 目标图像大小 (height, width)
            normalize: 是否进行标准化
            normalize_mean: 标准化均值
            normalize_std: 标准化标准差
            clip_range: 像素值剪切范围 (min, max)，通常为HU值范围
        """
        self.image_size = image_size
        self.normalize = normalize
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.clip_range = clip_range
    
    def __call__(
        self,
        image: np.ndarray,
        label: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        执行预处理
        
        Args:
            image: 输入图像
            label: 对应的标注（可选）
            
        Returns:
            处理后的 (image, label) 元组
        """
        # 像素值剪切
        if self.clip_range is not None:
            image = np.clip(image, self.clip_range[0], self.clip_range[1])
        
        # 调整大小
        image = self._resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        
        if label is not None:
            label = self._resize(label, self.image_size, interpolation=cv2.INTER_NEAREST)
        
        # 标准化
        if self.normalize:
            image = (image - self.normalize_mean) / self.normalize_std
        
        return image, label
    
    @staticmethod
    def _resize(
        image: np.ndarray,
        target_size: Tuple[int, int],
        interpolation: int = cv2.INTER_LINEAR,
    ) -> np.ndarray:
        """调整图像大小"""
        if image.shape[:2] == target_size:
            return image
        
        return cv2.resize(
            image,
            (target_size[1], target_size[0]),
            interpolation=interpolation
        )
    
    @staticmethod
    def crop_to_roi(
        image: np.ndarray,
        label: Optional[np.ndarray] = None,
        margin: int = 10,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        裁剪到感兴趣区域（ROI），去除过多背景
        
        Args:
            image: 输入图像
            label: 标注图像
            margin: 裁剪边界的边距
            
        Returns:
            裁剪后的 (image, label) 元组
        """
        if label is None:
            # 使用Otsu阈值自动检测前景
            _, binary = cv2.threshold(
                image.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        else:
            # 使用标注检测前景
            binary = (label > 0).astype(np.uint8) * 255
        
        # 找到前景边界
        coords = np.where(binary > 0)
        if len(coords[0]) == 0:
            return image, label
        
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        # 添加边距
        y_min = max(0, y_min - margin)
        y_max = min(image.shape[0], y_max + margin)
        x_min = max(0, x_min - margin)
        x_max = min(image.shape[1], x_max + margin)
        
        image_cropped = image[y_min:y_max, x_min:x_max]
        label_cropped = label[y_min:y_max, x_min:x_max] if label is not None else None
        
        return image_cropped, label_cropped
    
    @staticmethod
    def intensity_normalization(
        image: np.ndarray,
        method: str = "minmax",
    ) -> np.ndarray:
        """
        强度标准化
        
        Args:
            image: 输入图像
            method: 标准化方法 ('minmax', 'zscore', 'percentile')
            
        Returns:
            标准化后的图像
        """
        if method == "minmax":
            # 最小-最大标准化 [0, 1]
            min_val = image.min()
            max_val = image.max()
            if max_val - min_val == 0:
                return np.zeros_like(image)
            return (image - min_val) / (max_val - min_val)
        
        elif method == "zscore":
            # Z-score标准化
            mean = image.mean()
            std = image.std()
            if std == 0:
                return np.zeros_like(image)
            return (image - mean) / std
        
        elif method == "percentile":
            # 百分位数标准化
            p2, p98 = np.percentile(image, [2, 98])
            image_clipped = np.clip(image, p2, p98)
            return (image_clipped - p2) / (p98 - p2)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    @staticmethod
    def remove_small_objects(
        label: np.ndarray,
        min_size: int = 100,
    ) -> np.ndarray:
        """
        移除小的连通分量
        
        Args:
            label: 标注图像
            min_size: 最小连通分量大小
            
        Returns:
            处理后的标注
        """
        from scipy import ndimage
        
        label_processed = label.copy()
        
        for class_id in np.unique(label):
            if class_id == 0:  # 跳过背景
                continue
            
            # 获取当前类别的二值图像
            binary = (label == class_id).astype(np.uint8)
            
            # 标记连通分量
            labeled_array, num_features = ndimage.label(binary)
            
            # 移除小的连通分量
            for i in range(1, num_features + 1):
                size = np.sum(labeled_array == i)
                if size < min_size:
                    label_processed[labeled_array == i] = 0
        
        return label_processed
