# 数据准备指南

本文档详细说明了为Chinese-CLIP模型微调准备数据的完整流程。

## 1. 数据集结构

Chinese-CLIP微调数据集需要遵循特定的结构和格式要求。我们使用的三个数据集都按照以下格式组织：

```
dataset_name/
├── images/                # 图像文件夹
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── train.jsonl            # 训练集标注文件
├── val.jsonl              # 验证集标注文件
└── test.jsonl             # 测试集标注文件 (可选)
```

## 2. 标注文件格式

标注文件采用JSONL格式（每行一个JSON对象）。每个JSON对象包含以下字段：

```json
{
  "image_path": "images/img1.jpg",  
  "text": "这是图像的中文描述文本"
}
```

对于Flickr30k-CN等具有多个描述的数据集，每个图像会对应多个JSON条目。

## 3. MUGE数据集

### 3.1 获取途径

MUGE数据集可从清华大学开源：[https://www.luge.ai/#/luge/dataDetail?id=28](https://www.luge.ai/#/luge/dataDetail?id=28)

### 3.2 数据格式转换

MUGE原始格式需要转换为Chinese-CLIP训练格式：

```python
import json
import os

def convert_muge_to_clip_format(input_file, output_file, image_dir):
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            item = json.loads(line.strip())
            image_id = item['image_id']
            image_path = os.path.join(image_dir, f"{image_id}.jpg")
            
            # 确保图像存在
            if not os.path.exists(image_path):
                continue
                
            for text in item['text']:
                clip_item = {
                    "image_path": image_path,
                    "text": text
                }
                fout.write(json.dumps(clip_item, ensure_ascii=False) + '\n')

# 转换训练集
convert_muge_to_clip_format('muge_train.json', 'train.jsonl', 'images')
# 转换验证集
convert_muge_to_clip_format('muge_dev.json', 'val.jsonl', 'images')
```

### 3.3 LMDB格式转换

为提高训练效率，将数据集转换为LMDB格式：

```bash
cd Chinese-CLIP
python cn_clip/preprocess/preprocess_lmdb.py \
    --input_dataset_dir /path/to/MUGE \
    --output_lmdb_dir /path/to/MUGE/lmdb \
    --task image_text \
    --annotation_file train.jsonl
```

## 4. Flickr30k-CN数据集

### 4.1 获取途径

Flickr30k-CN数据集可从GitHub获取：[https://github.com/li-xirong/Flickr30K-CN](https://github.com/li-xirong/Flickr30K-CN)

### 4.2 数据处理

Flickr30k-CN的处理流程：

```python
import json
import os
import pandas as pd

def convert_flickr30k_cn_to_clip_format(caption_file, output_file, image_dir):
    df = pd.read_csv(caption_file, sep='\t', header=None)
    df.columns = ['image_id', 'caption_id', 'text']
    
    with open(output_file, 'w', encoding='utf-8') as fout:
        for _, row in df.iterrows():
            image_id = str(row['image_id'])
            image_path = os.path.join(image_dir, f"{image_id}.jpg")
            
            # 确保图像存在
            if not os.path.exists(image_path):
                continue
                
            clip_item = {
                "image_path": image_path,
                "text": row['text']
            }
            fout.write(json.dumps(clip_item, ensure_ascii=False) + '\n')

# 转换训练集
convert_flickr30k_cn_to_clip_format('flickr30k_cn_train.txt', 'train.jsonl', 'images')
# 转换验证集
convert_flickr30k_cn_to_clip_format('flickr30k_cn_val.txt', 'val.jsonl', 'images')
```

## 5. 私有数据集

### 5.1 收集与标注

我们收集了146张材料科学领域的图像，并请专业人员进行标注。标注过程：

1. **图像收集**：从专业论文、材料数据库和实验室拍摄中收集
2. **标注要求**：
   - 每张图像提供2-3个中文描述
   - 描述必须包含材料名称、形态特征和表面特性
   - 避免使用太笼统的描述
   - 确保专业术语的准确性

### 5.2 数据转换

我们使用了脚本`convert_private_data.py`将私有数据集转换为所需格式：

```bash
python model_training/scripts/convert_private_data.py \
    --input_image_dir /path/to/private_images \
    --input_annotation_file /path/to/annotations.csv \
    --output_dir /path/to/private_dataset \
    --split_ratio 0.8
```

### 5.3 数据增强

由于私有数据集规模较小，我们应用了更强的数据增强策略：

```python
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                         (0.26862954, 0.26130258, 0.27577711))
])
```

## 6. 数据质量控制

### 6.1 图像预处理

- **尺寸检查**：确保所有图像分辨率不低于224×224
- **格式统一**：统一转换为RGB模式的JPEG格式
- **损坏检测**：筛选并移除损坏的图像文件

### 6.2 文本预处理

- **长度限制**：确保所有文本长度不超过52个token
- **特殊字符处理**：删除或替换不必要的特殊字符
- **文本去重**：移除完全相同的重复描述

### 6.3 图文匹配验证

随机抽样检查图像与描述是否正确匹配，确保数据集的质量。

## 7. 数据分析

### 7.1 数据集统计

| 指标 | MUGE | Flickr30k-CN | 私有数据集 |
|------|------|-------------|-----------|
| 图像数量 | 330万 | 31,783 | 146 |
| 文本数量 | 330万 | 158,915 | 342 |
| 平均每图文本数 | 1 | 5 | 2.34 |
| 平均文本长度 | 24.7字 | 36.2字 | 19.8字 |

### 7.2 词频分析

在私有数据集中，最常见的专业术语词频统计：

| 术语 | 出现次数 | 占比 |
|------|---------|-----|
| 碳纤维 | 78 | 22.8% |
| 复合材料 | 65 | 19.0% |
| 聚合物 | 53 | 15.5% |
| 纳米材料 | 42 | 12.3% |
| 金属基 | 37 | 10.8% |

## 8. 注意事项

1. **版权问题**：确保数据集中的图像没有版权问题，或已获得授权
2. **隐私保护**：移除可能包含个人信息的图像
3. **存储空间**：MUGE完整数据集较大，请确保有足够空间(约100GB)
4. **预处理时间**：大型数据集转换为LMDB格式可能需要数小时

## 9. 参考资源

- [Chinese-CLIP官方数据格式说明](https://github.com/OFA-Sys/Chinese-CLIP/blob/master/README.md)
- [MUGE数据集官方页面](https://www.luge.ai/#/luge/dataDetail?id=28)
- [Flickr30k-CN数据集GitHub](https://github.com/li-xirong/Flickr30K-CN) 