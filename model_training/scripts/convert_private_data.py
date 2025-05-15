import os
import base64
import json
from io import BytesIO
from PIL import Image
import random
import sys

# 切换到项目根目录
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)

# 数据集路径
train_dir = "data/datasets/train"
val_dir = "data/datasets/val"
test_dir = "data/datasets/test"
output_dir = "data/private_dataset"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

def process_directory(dir_path, split):
    """处理指定目录，生成tsv和jsonl文件"""
    img_files = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.jpg') and file.startswith('private_'):
                img_files.append(os.path.join(root, file))
    
    # 创建输出文件
    tsv_path = os.path.join(output_dir, f"{split}_imgs.tsv")
    jsonl_path = os.path.join(output_dir, f"{split}_texts.jsonl")
    
    image_id_map = {}  # 文件名到ID的映射
    
    # 处理图片，创建tsv文件
    with open(tsv_path, 'w') as tsv_file:
        for i, img_path in enumerate(img_files):
            try:
                # 为每个图片分配一个唯一ID
                image_id = i + 1
                image_id_map[os.path.basename(img_path)] = image_id
                
                # 读取图片并转换为base64
                img = Image.open(img_path)
                img_buffer = BytesIO()
                img.save(img_buffer, format=img.format if img.format else 'JPEG')
                byte_data = img_buffer.getvalue()
                base64_str = base64.b64encode(byte_data).decode("utf-8")
                
                # 写入tsv文件
                tsv_file.write(f"{image_id}\t{base64_str}\n")
                
                print(f"Processed image {i+1}/{len(img_files)}: {img_path}")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    # 处理文本，创建jsonl文件
    with open(jsonl_path, 'w') as jsonl_file:
        text_id = 0
        for img_file in img_files:
            txt_file = img_file.replace('.jpg', '.txt')
            if os.path.exists(txt_file):
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    
                    img_filename = os.path.basename(img_file)
                    if img_filename in image_id_map:
                        text_id += 1
                        # 创建jsonl条目
                        entry = {
                            "text_id": text_id,
                            "text": text,
                            "image_ids": [image_id_map[img_filename]]
                        }
                        jsonl_file.write(json.dumps(entry, ensure_ascii=False) + '\n')
                except Exception as e:
                    print(f"Error processing {txt_file}: {e}")
    
    print(f"Created {split} files: {tsv_path} and {jsonl_path}")
    return len(img_files)

# 处理各个目录
print("处理训练集...")
train_count = process_directory(train_dir, "train")
print("处理验证集...")
val_count = process_directory(val_dir, "valid")
print("处理测试集...")
test_count = process_directory(test_dir, "test")

print(f"处理完成！共处理了 {train_count + val_count + test_count} 张图片")
print(f"- 训练集: {train_count} 张图片")
print(f"- 验证集: {val_count} 张图片")
print(f"- 测试集: {test_count} 张图片")
print(f"\n转换后的数据保存在 {output_dir} 目录下")
print("接下来可以使用Chinese-CLIP的build_lmdb_dataset.py脚本将数据转换为LMDB格式") 