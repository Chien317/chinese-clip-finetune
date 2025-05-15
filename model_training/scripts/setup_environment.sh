#!/bin/bash
# 云服务器环境设置脚本
# 用于设置Chinese-CLIP模型训练的运行环境

# 更新系统并安装必要依赖
apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    python3 \
    python3-pip \
    python3-venv

# 创建Python虚拟环境
python3 -m venv ~/.venv
source ~/.venv/bin/activate

# 安装PyTorch（CUDA支持版本）
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# 安装其他依赖
pip install \
    numpy \
    pandas \
    pillow \
    scikit-learn \
    matplotlib \
    tqdm \
    lmdb \
    ftfy \
    regex \
    sentencepiece \
    transformers==4.15.0 \
    timm==0.5.4 \
    jsonlines

# 克隆Chinese-CLIP代码仓库（如果尚未克隆）
if [ ! -d "Chinese-CLIP" ]; then
    git clone https://github.com/OFA-Sys/Chinese-CLIP.git
    cd Chinese-CLIP
    pip install -e .
    cd ..
fi

# 创建数据和模型目录
mkdir -p data/MUGE
mkdir -p data/Flickr30k-CN
mkdir -p data/private_dataset/lmdb
mkdir -p models/pretrained_weights
mkdir -p experiments

echo "环境设置完成！"
echo "请使用 'source ~/.venv/bin/activate' 激活虚拟环境"
