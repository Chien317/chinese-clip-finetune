#!/bin/bash

# 检查训练状态的脚本
# 用于获取当前训练进度和资源使用情况的摘要

# 切换到项目根目录
cd $(dirname $0)/..
WORKDIR=$(pwd)
echo "当前工作目录: $WORKDIR"

# 标题栏
echo "==================================================="
echo "        Chinese-CLIP 训练状态检查工具               "
echo "==================================================="
echo

# 检查系统负载
echo "系统负载:"
uptime
echo

# 检查GPU状态
echo "GPU状态:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "未找到 nvidia-smi 命令，无法获取GPU信息"
fi
echo

# 检查磁盘使用情况
echo "磁盘使用情况:"
df -h .
echo

# 检查训练进程
echo "训练相关进程:"
ps aux | grep -E "train|python" | grep -v grep
echo

# 检查最新的训练日志
echo "最近的训练日志:"
if [ -f "logs/training.log" ]; then
    tail -n 20 logs/training.log
else
    echo "未找到训练日志文件"
fi
echo

# 检查训练结果目录
echo "训练结果文件:"
if [ -d "experiments" ]; then
    find experiments -name "*.pt" -type f -printf "%TY-%Tm-%Td %TH:%TM:%TS %p\n" 2>/dev/null | sort
else
    echo "未找到训练结果目录"
fi
echo

# 输出完成的训练阶段
echo "已完成的训练阶段:"
completed_stages=0

if [ -d "experiments/muge_finetune_vit-b-16_roberta-base_bs128_8gpu" ]; then
    echo "✓ MUGE数据集训练阶段已完成"
    completed_stages=$((completed_stages+1))
else
    echo "✗ MUGE数据集训练阶段未完成"
fi

if [ -d "experiments/flickr30k_finetune_vit-b-16_roberta-base_bs128_8gpu" ]; then
    echo "✓ Flickr30k-CN数据集训练阶段已完成"
    completed_stages=$((completed_stages+1))
else
    echo "✗ Flickr30k-CN数据集训练阶段未完成"
fi

if [ -d "experiments/private_finetune_from_muge" ]; then
    echo "✓ MUGE+私有数据微调阶段已完成"
    completed_stages=$((completed_stages+1))
else
    echo "✗ MUGE+私有数据微调阶段未完成"
fi

if [ -d "experiments/private_finetune_from_flickr" ]; then
    echo "✓ Flickr30k-CN+私有数据微调阶段已完成"
    completed_stages=$((completed_stages+1))
else
    echo "✗ Flickr30k-CN+私有数据微调阶段未完成"
fi

echo
echo "训练完成度: $completed_stages/4 阶段"
echo "===================================================" 