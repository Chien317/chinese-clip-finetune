# Chinese-CLIP 云服务器训练指南

本文档提供了在Alibaba Cloud服务器上设置和运行Chinese-CLIP模型训练的完整步骤。

## 服务器环境

- **服务器IP**: 120.77.9.148
- **GPU**: Tesla V100
- **连接方式**: SSH密钥认证

## 设置步骤

### 1. 连接服务器

使用SSH密钥连接服务器：

```bash
ssh root@120.77.9.148
```

### 2. 克隆代码仓库

```bash
git clone https://github.com/Chien317/chinese-clip-finetune.git
cd chinese-clip-finetune
```

### 3. 设置环境

我们提供了一个自动化脚本来设置所有必要的环境：

```bash
bash model_training/scripts/setup_environment.sh
```

这个脚本会：
- 安装系统依赖
- 创建Python虚拟环境
- 安装PyTorch (CUDA版本)和其他依赖
- 克隆Chinese-CLIP代码仓库
- 创建必要的数据和模型目录

### 4. 下载预训练模型

```bash
bash model_training/scripts/download_models.sh
```

### 5. 下载和准备数据集

```bash
bash model_training/scripts/download_datasets.sh
```

这个脚本会下载：
- MUGE数据集 (~2.5GB)
- Flickr30k-CN数据集 (~2.5GB)
并将它们转换为LMDB格式以加速训练。

## 运行训练流程

### 启动完整训练流水线

我们的训练流水线包括三个阶段：
1. 使用MUGE数据集微调预训练模型
2. 使用Flickr30k-CN数据集微调模型
3. 使用私有标注数据集进行最终微调

启动训练（可选邮件通知）：

```bash
# 不需要邮件通知
bash model_training/scripts/start_training.sh

# 或者，训练完成后发送邮件通知
bash model_training/scripts/start_training.sh your-email@example.com
```

### 监控训练进度

训练日志将保存在`logs/`目录下：

```bash
# 查看训练日志
tail -f logs/training.log

# 查看GPU使用情况记录
tail -f logs/gpu_usage.log

# 查看训练流水线详细输出
tail -f logs/train_pipeline.log
```

## 训练结果

所有训练结果将保存在`experiments/`目录下：

```
experiments/
├── muge_finetune_vit-b-16_roberta-base_bs128_8gpu/     # MUGE数据集训练结果
├── flickr30k_finetune_vit-b-16_roberta-base_bs128_8gpu/ # Flickr30k-CN数据集训练结果
├── private_finetune_from_muge/                         # MUGE+私有数据训练结果
└── private_finetune_from_flickr/                       # Flickr30k-CN+私有数据训练结果
```

最终模型将以`.pt`格式保存，可以直接用于推理或Web应用。

## 常见问题解决

### 训练中断

训练脚本设计有自动重试机制，最多尝试3次。如果仍然失败，可以手动重新启动：

```bash
bash model_training/scripts/start_training.sh
```

### 显存不足

如果遇到显存不足的问题，可以尝试减小批处理大小：

```bash
# 编辑训练流水线脚本
vim model_training/scripts/train_pipeline.sh

# 将batch_size从32改为16或更小
```

### 手动运行单个阶段

如果需要单独运行某个训练阶段：

```bash
# 只运行MUGE数据集训练
bash Chinese-CLIP/run_scripts/muge_finetune_vit-b-16_rbt-base.sh .

# 只运行私有数据集微调
bash model_training/scripts/private_finetune_vit-b-16_rbt-base.sh .
``` 