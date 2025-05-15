#!/bin/bash

# 数据集下载和提取脚本
# 用于下载MUGE和Flickr30k-CN数据集，并准备用于训练

# 切换到项目根目录
cd $(dirname $0)/..
WORKDIR=$(pwd)
echo "当前工作目录: $WORKDIR"

# 创建数据目录
mkdir -p data/MUGE
mkdir -p data/Flickr30k-CN
mkdir -p data/private_dataset

# 下载MUGE数据集
echo "==== 下载MUGE数据集 ===="
cd data/MUGE
wget https://ali-ai-tron.oss-cn-zhangjiakou.aliyuncs.com/muge_intro_2021/muge_train_text.jsonl
wget https://ali-ai-tron.oss-cn-zhangjiakou.aliyuncs.com/muge_intro_2021/muge_dev_text.jsonl
wget https://ali-ai-tron.oss-cn-zhangjiakou.aliyuncs.com/muge_intro_2021/muge_test_text.jsonl
wget https://ali-ai-tron.oss-cn-zhangjiakou.aliyuncs.com/muge_intro_2021/muge_train_imgs.tsv
wget https://ali-ai-tron.oss-cn-zhangjiakou.aliyuncs.com/muge_intro_2021/muge_dev_imgs.tsv
wget https://ali-ai-tron.oss-cn-zhangjiakou.aliyuncs.com/muge_intro_2021/muge_test_imgs.tsv

# 下载Flickr30k-CN数据集
echo "==== 下载Flickr30k-CN数据集 ===="
cd $WORKDIR/data/Flickr30k-CN
wget https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/datasets/flickr30k/train_imgs.tsv
wget https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/datasets/flickr30k/valid_imgs.tsv
wget https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/datasets/flickr30k/test_imgs.tsv
wget https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/datasets/flickr30k/train_texts.jsonl
wget https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/datasets/flickr30k/valid_texts.jsonl
wget https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/datasets/flickr30k/test_texts.jsonl

# 创建LMDB文件以加速训练
echo "==== 为MUGE数据集创建LMDB文件 ===="
cd $WORKDIR
python Chinese-CLIP/cn_clip/preprocessing/build_lmdb.py \
    --dataset MUGE \
    --image-data data/MUGE/muge_train_imgs.tsv \
    --text-data data/MUGE/muge_train_text.jsonl \
    --output data/MUGE/lmdb/train

python Chinese-CLIP/cn_clip/preprocessing/build_lmdb.py \
    --dataset MUGE \
    --image-data data/MUGE/muge_dev_imgs.tsv \
    --text-data data/MUGE/muge_dev_text.jsonl \
    --output data/MUGE/lmdb/valid

python Chinese-CLIP/cn_clip/preprocessing/build_lmdb.py \
    --dataset MUGE \
    --image-data data/MUGE/muge_test_imgs.tsv \
    --text-data data/MUGE/muge_test_text.jsonl \
    --output data/MUGE/lmdb/test

echo "==== 为Flickr30k-CN数据集创建LMDB文件 ===="
python Chinese-CLIP/cn_clip/preprocessing/build_lmdb.py \
    --dataset Flickr30k \
    --image-data data/Flickr30k-CN/train_imgs.tsv \
    --text-data data/Flickr30k-CN/train_texts.jsonl \
    --output data/Flickr30k-CN/lmdb/train

python Chinese-CLIP/cn_clip/preprocessing/build_lmdb.py \
    --dataset Flickr30k \
    --image-data data/Flickr30k-CN/valid_imgs.tsv \
    --text-data data/Flickr30k-CN/valid_texts.jsonl \
    --output data/Flickr30k-CN/lmdb/valid

python Chinese-CLIP/cn_clip/preprocessing/build_lmdb.py \
    --dataset Flickr30k \
    --image-data data/Flickr30k-CN/test_imgs.tsv \
    --text-data data/Flickr30k-CN/test_texts.jsonl \
    --output data/Flickr30k-CN/lmdb/test

echo "数据集下载和处理完成!" 