#!/bin/bash

# 云服务器训练脚本
# 此脚本用于在云服务器上下载预处理的数据集并执行训练

# 切换到项目根目录
cd /root/chinese-clip-finetune/model_training || {
  echo "项目目录不存在，尝试克隆代码..."
  cd /root
  git clone https://github.com/Chien317/chinese-clip-finetune.git
  cd chinese-clip-finetune/model_training
}

# 确保环境已设置
if [ ! -d "venv" ]; then
  echo "设置环境..."
  bash scripts/setup_environment.sh
fi

# 激活环境
source venv/bin/activate

# 下载预处理的数据集
echo "下载预处理数据集..."
mkdir -p data/private_dataset/lmdb/train data/private_dataset/lmdb/valid

# 下载预处理数据（替换为您的GitHub仓库URL）
REPO_URL="https://github.com/yourusername/chinese-clip-dataset"
TMP_DIR="/tmp/dataset_download"

rm -rf $TMP_DIR
mkdir -p $TMP_DIR
cd $TMP_DIR || exit 1

git clone $REPO_URL .
cp -r private_data/* /root/chinese-clip-finetune/model_training/data/private_dataset/

# 切回项目目录
cd /root/chinese-clip-finetune/model_training || exit 1

# 下载预训练模型（如果尚未下载）
if [ ! -f "models/pretrained_weights/clip_cn_vit-b-16.pt" ]; then
  echo "下载预训练模型..."
  bash scripts/download_models.sh
fi

# 将数据转换为LMDB格式
echo "将数据转换为LMDB格式..."
cd Chinese-CLIP || exit 1

# 训练集
python cn_clip/clip/build_lmdb_dataset.py \
  --tsv_path /root/chinese-clip-finetune/model_training/data/private_dataset/train_imgs.tsv \
  --jsonl_path /root/chinese-clip-finetune/model_training/data/private_dataset/train_texts.jsonl \
  --output_path /root/chinese-clip-finetune/model_training/data/private_dataset/lmdb/train

# 验证集
python cn_clip/clip/build_lmdb_dataset.py \
  --tsv_path /root/chinese-clip-finetune/model_training/data/private_dataset/valid_imgs.tsv \
  --jsonl_path /root/chinese-clip-finetune/model_training/data/private_dataset/valid_texts.jsonl \
  --output_path /root/chinese-clip-finetune/model_training/data/private_dataset/lmdb/valid

cd ..

# 开始训练流程
echo "开始训练流程..."
bash scripts/train_pipeline.sh

echo "训练完成，查看 experiments/ 目录下的结果" 