"""
数据增强模块
包含随机旋转、翻转、弹性变形、高斯噪声等增强操作
"""

import numpy as np
import cv2
import albumentations as A
from typing import Tuple, Optional, Dict


class CardiacAugmentation:
    """心脏MRI图像数据增强"""
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        rotation_angle: int = 15,
        flip_prob: float = 0.5,
        elastic_deformation: float = 0.3,
        gaussian_noise_std: float = 0.01,
        brightness_contrast: float = 0.2,
    ):
        """
        初始化数据增强
        
        Args:
            config: 配置字典（优先使用）
            rotation_angle: 旋转角度范围
            flip_prob: 翻转概率
            elastic_deformation: 弹性变形强度
            gaussian_noise_std: 高斯噪声标准差
            brightness_contrast: 亮度对比度变化范围
        """
        if config is not None:
            self.rotation_angle = config.get('rotation_angle', rotation_angle)
            self.flip_prob = config.get('flip_prob', flip_prob)
            self.elastic_deformation = config.get('elastic_deformation', elastic_deformation)
            self.gaussian_noise_std = config.get('gaussian_noise_std', gaussian_noise_std)
            self.brightness_contrast = config.get('brightness_contrast', brightness_contrast)
        else:
            self.rotation_angle = rotation_angle
            self.flip_prob = flip_prob
            self.elastic_deformation = elastic_deformation
            self.gaussian_noise_std = gaussian_noise_std
            self.brightness_contrast = brightness_contrast
        
        self.transform = self._build_augmentation()
    
    def _build_augmentation(self) -> A.Compose:
        """构建Albumentations增强管道"""
        transforms = [
            A.Rotate(
                limit=self.rotation_angle,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT,
                p=0.8,
            ),
            A.HorizontalFlip(p=self.flip_prob),
            A.VerticalFlip(p=self.flip_prob),
            A.ElasticTransform(
                alpha=76,
                sigma=self.elastic_deformation * 100,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT,
                p=0.5,
            ),
            A.GaussNoise(p=0.3, per_channel=False),
            A.RandomBrightnessContrast(
                brightness_limit=self.brightness_contrast,
                contrast_limit=self.brightness_contrast,
                p=0.5,
            ),
            A.GaussianBlur(blur_limit=3, p=0.2),
        ]
        
        return A.Compose(
            transforms,
            additional_targets={'mask': 'mask'}
        )
    
    def __call__(
        self,
        image: np.ndarray,
        label: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行数据增强
        
        Args:
            image: 输入图像 (H, W) 范围 [0, 1]
            label: 标注图像 (H, W)
            
        Returns:
            增强后的 (image, label) 元组
        """
        # 检查输入范围
        image = np.clip(image, 0, 1).astype(np.float32)
        label = label.astype(np.uint8)
        
        # 应用Albumentations增强
        augmented = self.transform(image=image, mask=label)
        image_aug = augmented['image'].astype(np.float32)
        label_aug = augmented['mask'].astype(np.int64)
        
        return image_aug, label_aug


class RandomAugmentation:
    """随机单个增强操作"""
    
    @staticmethod
    def random_rotation(
        image: np.ndarray,
        label: np.ndarray,
        angle_range: Tuple[int, int] = (-15, 15),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """随机旋转"""
        angle = np.random.uniform(angle_range[0], angle_range[1])
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        image_rotated = cv2.warpAffine(
            image, matrix, (w, h),
            borderMode=cv2.BORDER_REFLECT
        )
        label_rotated = cv2.warpAffine(
            label, matrix, (w, h),
            borderMode=cv2.BORDER_REFLECT,
            flags=cv2.INTER_NEAREST
        )
        
        return image_rotated, label_rotated
    
    @staticmethod
    def random_flip(
        image: np.ndarray,
        label: np.ndarray,
        axis: int = 1,  # 0: vertical, 1: horizontal
    ) -> Tuple[np.ndarray, np.ndarray]:
        """随机翻转"""
        if np.random.rand() > 0.5:
            image = cv2.flip(image, axis)
            label = cv2.flip(label, axis)
        
        return image, label
    
    @staticmethod
    def random_elastic_deformation(
        image: np.ndarray,
        label: np.ndarray,
        alpha: float = 76,
        sigma: float = 30,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """随机弹性变形"""
        h, w = image.shape[:2]
        
        # 生成随机位移场
        random_state = np.random.RandomState(None)
        dx = random_state.randn(h, w) * sigma
        dy = random_state.randn(h, w) * sigma
        
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        
        image_deformed = cv2.remap(
            image, indices[1].reshape(h, w).astype(np.float32),
            indices[0].reshape(h, w).astype(np.float32),
            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )
        
        label_deformed = cv2.remap(
            label, indices[1].reshape(h, w).astype(np.float32),
            indices[0].reshape(h, w).astype(np.float32),
            cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT
        )
        
        return image_deformed, label_deformed
    
    @staticmethod
    def add_gaussian_noise(
        image: np.ndarray,
        std: float = 0.01,
    ) -> np.ndarray:
        """添加高斯噪声"""
        noise = np.random.randn(*image.shape) * std
        image_noisy = image + noise
        return np.clip(image_noisy, 0, 1)
    
    @staticmethod
    def random_brightness_contrast(
        image: np.ndarray,
        brightness_limit: float = 0.2,
        contrast_limit: float = 0.2,
    ) -> np.ndarray:
        """随机调整亮度和对比度"""
        brightness = 1 + np.random.uniform(-brightness_limit, brightness_limit)
        contrast = 1 + np.random.uniform(-contrast_limit, contrast_limit)
        
        image_aug = image * contrast + brightness
        return np.clip(image_aug, 0, 1)
    
    @staticmethod
    def random_gaussian_blur(
        image: np.ndarray,
        kernel_size: int = 3,
    ) -> np.ndarray:
        """随机高斯模糊"""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


class MixupAugmentation:
    """Mixup数据增强：线性插值混合两个样本"""
    
    @staticmethod
    def mixup(
        image1: np.ndarray,
        label1: np.ndarray,
        image2: np.ndarray,
        label2: np.ndarray,
        alpha: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mixup增强
        
        Args:
            image1, label1: 第一个样本
            image2, label2: 第二个样本
            alpha: 混合权重
            
        Returns:
            混合后的 (image, label) 元组
        """
        # 混合图像
        image_mixed = alpha * image1 + (1 - alpha) * image2
        
        # 混合标注（取概率较高的类别）
        if alpha >= 0.5:
            label_mixed = label1
        else:
            label_mixed = label2
        
        return image_mixed, label_mixed
