# Chinese-CLIP 图文对齐模型微调项目

一个基于Chinese-CLIP模型的图文对齐微调项目，专注于中文环境下的多模态学习。本项目包含完整的模型微调流程、云服务器训练配置和应用集成方案。

## 项目概述

本项目通过微调Chinese-CLIP模型来优化中文图文对齐能力，主要用于以下场景：
- 中文环境下的图像搜索
- 基于文本的图像检索
- 基于图像的文本匹配
- 多模态内容理解与推荐

## 微调策略

项目采用了三阶段微调策略：
1. **使用MUGE数据集进行初步微调**：适应中文通用语义
2. **使用Flickr30k-CN数据集进行进一步微调**：提升中文图文对齐能力
3. **使用私有数据集进行精细微调**：适应特定领域需求

## 项目结构

```
chinese-clip-finetune/
├── model_training/           # 模型训练相关代码和数据
│   ├── scripts/              # 训练和数据处理脚本
│   │   ├── convert_private_data.py   # 私有数据转换脚本
│   │   ├── train_pipeline.sh         # 完整训练流水线脚本
│   │   ├── setup_environment.sh      # 环境设置脚本
│   │   ├── download_models.sh        # 下载预训练模型脚本
│   │   ├── download_datasets.sh      # 下载数据集脚本
│   │   ├── start_training.sh         # 启动训练脚本
│   │   └── check_status.sh           # 检查训练状态脚本
│   ├── docs/                 # 训练相关文档
│   │   └── README_finetune.md # 微调策略说明文档
│   └── CLOUD_README.md       # 云服务器训练指南
├── models/                   # 模型文件目录
│   └── pretrained_weights/   # 预训练权重（需要手动下载）
├── tools/                    # 辅助工具
│   └── model_convert.py      # 模型格式转换工具
└── README.md                 # 主文档
```

## 使用指南

### 在云服务器上训练模型

1. **连接服务器并克隆仓库**
```bash
git clone https://github.com/Chien317/chinese-clip-finetune.git
cd chinese-clip-finetune
```

2. **设置环境**
```bash
bash model_training/scripts/setup_environment.sh
```

3. **下载预训练模型**
```bash
bash model_training/scripts/download_models.sh
```

4. **下载和准备数据集**
```bash
bash model_training/scripts/download_datasets.sh
```

5. **启动训练流程**
```bash
bash model_training/scripts/start_training.sh
```

6. **检查训练状态**
```bash
bash model_training/scripts/check_status.sh
```

### 使用你自己的数据集进行微调

1. **准备数据集**：按照格式要求准备图像和文本描述
2. **数据转换**：使用`convert_private_data.py`脚本转换数据格式
3. **启动微调**：修改`train_pipeline.sh`中的参数，指定你的数据集路径
4. **运行训练**：执行上述训练流程

## 训练结果

训练完成后，模型将保存在`experiments/`目录下，可以选择性能最好的模型用于应用部署。

## 技术栈

- **基础模型**：Chinese-CLIP (ViT-B/16 + RoBERTa-wwm-ext-Base)
- **训练框架**：PyTorch
- **训练数据集**：MUGE、Flickr30k-CN、私有数据集
- **训练环境**：CUDA + Tesla V100 GPU

## 常见问题解决

请参阅 [云服务器训练指南](model_training/CLOUD_README.md) 中的常见问题解决方案部分。

## 关于Chinese-CLIP

Chinese-CLIP是针对中文环境优化的多模态预训练模型，由OFA团队开发。本项目基于Chinese-CLIP进行微调，以适应特定应用场景的需求。

## 许可

MIT License 