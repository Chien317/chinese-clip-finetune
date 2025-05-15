#!/bin/bash
# 脚本用于下载预训练模型

mkdir -p ../models/pretrained_weights
cd ../models/pretrained_weights

# 下载Chinese-CLIP预训练权重
echo "下载Chinese-CLIP ViT-B-16 预训练权重..."
# Mac使用curl替代wget
curl -O https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/clip_cn_vit-b-16.pt

echo "预训练模型下载完成!"
