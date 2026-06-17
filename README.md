# 心脏磁共振影像心腔结构自动分割

这是一个基于深度学习的心脏MRI自动分割项目，使用M&Ms数据集进行训练和评估。

## 项目概述

- **任务**: 心腔结构自动分割（医学影像分割）
- **数据集**: M&Ms (Multi-Centre, Multi-Vendor & Multi-Disease)
- **目标**: 分割左心室(LV)、右心室(RV)和心肌(MYO)
- **模型**: 基于U-Net和ResNet的多阶段分割网络
- **评估指标**: Dice系数、IoU、Hausdorff距离等

## 项目结构

```
cardiac-mri-segmentation/
├── README.md                 # 项目说明
├── requirements.txt          # 依赖包
├── config.yaml              # 配置文件
│
├── data/                    # 数据处理模块
│   ├── __init__.py
│   ├── dataset.py           # M&Ms数据集加载器
│   ├── preprocessor.py      # 数据预处理
│   └── augmentation.py      # 数据增强
│
├── models/                  # 模型模块
│   ├── __init__.py
│   ├── unet.py             # U-Net模型
│   ├── resnet.py           # ResNet编码器
│   └── losses.py           # 损失函数
│
├── train.py                # 训练脚本
├── infer.py                # 推理脚本
├── evaluate.py             # 评估脚本
│
└── utils/                  # 工具函数
    ├── __init__.py
    ├── metrics.py          # 评估指标
    └── visualization.py    # 可视化工具
```

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 准备数据
下载M&Ms数据集：https://www.ub.edu/mnms/

### 3. 训练模型
```bash
python train.py --config config.yaml
```

### 4. 推理测试
```bash
python infer.py --model_path checkpoints/best_model.pth --image_path test_image.nii.gz
```

### 5. 模型评估
```bash
python evaluate.py --model_path checkpoints/best_model.pth
```

## 关键特性

✅ **M&Ms数据集支持** - 完整的数据加载和预处理  
✅ **多阶段分割** - 基于最新研究的高精度分割  
✅ **丰富的数据增强** - 旋转、翻转、弹性变形等  
✅ **多种评估指标** - Dice、IoU、Hausdorff距离等  
✅ **易用的API** - 简单的训练和推理接口  
✅ **可视化工具** - 分割结果可视化  

## 性能指标

- **Left Ventricle (LV)**: Dice ≈ 0.95+
- **Right Ventricle (RV)**: Dice ≈ 0.92+
- **Myocardium (MYO)**: Dice ≈ 0.90+

## 参考文献

1. M&Ms Challenge: https://www.ub.edu/mnms/
2. [2412.09386] Multi-Stage Segmentation and Cascade Classification
3. CardSegNet: An adaptive hybrid CNN-vision transformer model for heart

## 许可证

MIT License
