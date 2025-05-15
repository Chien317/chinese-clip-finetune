#!/bin/bash

# 训练流水线脚本
# 策略：先用大数据集训练，再用私有数据集微调

# 切换到项目根目录
cd $(dirname $0)/..
WORKDIR=$(pwd)
echo "当前工作目录: $WORKDIR"

# 创建输出目录
mkdir -p experiments

# 为了解决导入错误，先执行修复代码
cat > ${WORKDIR}/fix_cn_clip_import.py << 'EOF'
import os
import sys

# 找到cn_clip模块路径
for path in sys.path:
    model_path = os.path.join(path, 'cn_clip/clip/model.py')
    if os.path.exists(model_path):
        # 检查文件内容
        with open(model_path, 'r') as f:
            content = f.read()
        
        # 如果没有convert_state_dict函数，添加它
        if 'def convert_state_dict' not in content:
            with open(model_path, 'a') as f:
                f.write('\n\ndef convert_state_dict(state_dict):\n')
                f.write('    """Convert state dict to be compatible with the model."""\n')
                f.write('    return {k.replace("module.", ""): v for k, v in state_dict.items()}\n')
            print(f"Added convert_state_dict function to {model_path}")
        else:
            print(f"convert_state_dict function already exists in {model_path}")
        break
EOF

# 执行修复脚本
python ${WORKDIR}/fix_cn_clip_import.py

# 替换Chinese-CLIP脚本中的torch.distributed.launch为torch.distributed.run
for script in Chinese-CLIP/run_scripts/*.sh; do
  if [ -f "$script" ]; then
    echo "修改脚本: $script"
    # 备份原始脚本
    cp "$script" "${script}.bak"
    # 替换launch为run
    sed -i '' 's/torch.distributed.launch/torch.distributed.run/g' "$script"
    # 替换--use_env为空（因为run默认使用环境变量）
    sed -i '' 's/--use_env//g' "$script"
  fi
done

# 第1步：用MUGE数据集微调预训练模型
echo "==== 第1步：使用MUGE数据集微调预训练模型 ===="
bash Chinese-CLIP/run_scripts/muge_finetune_vit-b-16_rbt-base.sh .
echo "MUGE数据集训练完成"

# 查找MUGE训练后的模型
MUGE_MODEL=$(find experiments/muge_finetune_vit-b-16_roberta-base_bs128_8gpu/ -name "epoch*.pt" 2>/dev/null | sort -V | tail -n 1)
if [ -z "$MUGE_MODEL" ]; then
  echo "未找到MUGE训练后的模型，将使用预训练模型继续"
  MUGE_MODEL="${WORKDIR}/models/pretrained_weights/clip_cn_vit-b-16.pt"
fi
echo "MUGE训练后的模型: $MUGE_MODEL"

# 第2步：用Flickr30k-CN数据集微调预训练模型
echo "==== 第2步：使用Flickr30k-CN数据集微调预训练模型 ===="
bash Chinese-CLIP/run_scripts/flickr30k_finetune_vit-b-16_rbt-base.sh .
echo "Flickr30k-CN数据集训练完成"

# 查找Flickr30k-CN训练后的模型
FLICKR_MODEL=$(find experiments/flickr30k_finetune_vit-b-16_roberta-base_bs128_8gpu/ -name "epoch*.pt" 2>/dev/null | sort -V | tail -n 1)
if [ -z "$FLICKR_MODEL" ]; then
  echo "未找到Flickr30k-CN训练后的模型，将使用预训练模型继续"
  FLICKR_MODEL="${WORKDIR}/models/pretrained_weights/clip_cn_vit-b-16.pt"
fi
echo "Flickr30k-CN训练后的模型: $FLICKR_MODEL"

# 创建私有数据微调脚本
cat > ${WORKDIR}/scripts/private_finetune_vit-b-16_rbt-base.sh << 'EOF'
#!/usr/bin/env

# 切换到项目根目录
cd $(dirname $0)/..
WORKDIR=$(pwd)

# 环境变量
export PYTHONPATH=${PYTHONPATH}:${WORKDIR}/Chinese-CLIP/

# 参数
DATAPATH=${1:-${WORKDIR}}
train_data=${DATAPATH}/data/private_dataset/lmdb/train
val_data=${DATAPATH}/data/private_dataset/lmdb/valid
resume=${DATAPATH}/models/pretrained_weights/clip_cn_vit-b-16.pt
output_base_dir=${DATAPATH}/experiments/
name=private_finetune_vit-b-16_roberta-base
batch_size=32
accum_freq=4
lr=5e-5
wd=0.001
max_epochs=20
context_length=52
warmup=100
vision_model=ViT-B-16
text_model=RoBERTa-wwm-ext-base-chinese

# 确保修复函数存在
python ${WORKDIR}/fix_cn_clip_import.py

# 使用Python直接运行
python ${WORKDIR}/Chinese-CLIP/cn_clip/training/main.py \
    --train-data=${train_data} \
    --val-data=${val_data} \
    --resume=${resume} \
    --reset-data-offset \
    --reset-optimizer \
    --logs=${output_base_dir} \
    --name=${name} \
    --log-interval=1 \
    --report-training-batch-acc \
    --context-length=${context_length} \
    --warmup=${warmup} \
    --batch-size=${batch_size} \
    --valid-batch-size=${batch_size} \
    --accum-freq=${accum_freq} \
    --lr=${lr} \
    --wd=${wd} \
    --max-epochs=${max_epochs} \
    --vision-model=${vision_model} \
    --use-augment \
    --text-model=${text_model}
EOF

chmod +x ${WORKDIR}/scripts/private_finetune_vit-b-16_rbt-base.sh

# 第3步：使用MUGE模型为基础，进一步用私有数据集微调
echo "==== 第3步：使用MUGE模型为基础，进一步用私有数据集微调 ===="

# 修改私有数据微调脚本以使用MUGE模型
cp scripts/private_finetune_vit-b-16_rbt-base.sh scripts/private_finetune_from_muge.sh
sed -i '' "s|resume=.*|resume=$MUGE_MODEL # MUGE trained model|" scripts/private_finetune_from_muge.sh
sed -i '' "s|name=.*|name=private_finetune_from_muge|" scripts/private_finetune_from_muge.sh

# 执行私有数据微调（基于MUGE模型）
bash scripts/private_finetune_from_muge.sh .
echo "基于MUGE模型的私有数据微调完成"

# 第4步：使用Flickr30k-CN模型为基础，进一步用私有数据集微调
echo "==== 第4步：使用Flickr30k-CN模型为基础，进一步用私有数据集微调 ===="

# 修改私有数据微调脚本以使用Flickr30k-CN模型
cp scripts/private_finetune_vit-b-16_rbt-base.sh scripts/private_finetune_from_flickr.sh
sed -i '' "s|resume=.*|resume=$FLICKR_MODEL # Flickr30k-CN trained model|" scripts/private_finetune_from_flickr.sh
sed -i '' "s|name=.*|name=private_finetune_from_flickr|" scripts/private_finetune_from_flickr.sh

# 执行私有数据微调（基于Flickr30k-CN模型）
bash scripts/private_finetune_from_flickr.sh .
echo "基于Flickr30k-CN模型的私有数据微调完成"

echo "==== 训练流水线执行完毕 ===="
echo "请从以下目录查看和比较训练模型的效果："
echo "1. MUGE微调模型：experiments/muge_finetune_vit-b-16_roberta-base_bs128_8gpu/"
echo "2. Flickr30k-CN微调模型：experiments/flickr30k_finetune_vit-b-16_roberta-base_bs128_8gpu/"
echo "3. MUGE+私有数据微调模型：experiments/private_finetune_from_muge/"
echo "4. Flickr30k-CN+私有数据微调模型：experiments/private_finetune_from_flickr/" 