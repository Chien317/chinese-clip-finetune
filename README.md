# 中文图文对齐模型训练数据集

这个仓库包含了用于训练Chinese-CLIP模型的预处理数据集。

## 数据集结构

```
.
├── private_data/
│   ├── train_imgs.tsv      # 训练集图像数据（base64编码）
│   ├── train_texts.jsonl   # 训练集文本数据
│   ├── valid_imgs.tsv      # 验证集图像数据
│   ├── valid_texts.jsonl   # 验证集文本数据
│   ├── test_imgs.tsv       # 测试集图像数据
│   └── test_texts.jsonl    # 测试集文本数据
└── README.md               # 说明文档
```

## 使用方法

### 1. 克隆仓库

```bash
git clone https://github.com/yourusername/chinese-clip-dataset.git
cd chinese-clip-dataset
```

### 2. 转换为LMDB格式

使用Chinese-CLIP的build_lmdb_dataset.py脚本将数据转换为LMDB格式：

```bash
# 在Chinese-CLIP目录下执行
cd Chinese-CLIP

# 训练集
python cn_clip/clip/build_lmdb_dataset.py \
  --tsv_path /path/to/private_data/train_imgs.tsv \
  --jsonl_path /path/to/private_data/train_texts.jsonl \
  --output_path /path/to/private_data/lmdb/train

# 验证集
python cn_clip/clip/build_lmdb_dataset.py \
  --tsv_path /path/to/private_data/valid_imgs.tsv \
  --jsonl_path /path/to/private_data/valid_texts.jsonl \
  --output_path /path/to/private_data/lmdb/valid
```

### 3. 训练模型

在转换完成后，可以使用Chinese-CLIP的训练脚本进行模型训练：

```bash
# 微调预训练模型
python cn_clip/training/main.py \
  --train-data=/path/to/private_data/lmdb/train \
  --val-data=/path/to/private_data/lmdb/valid \
  --resume=/path/to/pretrained_weights/clip_cn_vit-b-16.pt \
  --logs=./experiments \
  --name=finetune_private \
  --batch-size=32 \
  --context-length=52 \
  --warmup=100 \
  --lr=5e-5 \
  --wd=0.001 \
  --max-epochs=20 \
  --vision-model=ViT-B-16 \
  --text-model=RoBERTa-wwm-ext-base-chinese
```

## 数据集统计

- 训练集：100张图片及对应的文本描述
- 验证集：20张图片及对应的文本描述
- 测试集：20张图片及对应的文本描述

## 许可

私有数据集，仅限研究和学习使用。 