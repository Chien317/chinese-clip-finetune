#!/usr/bin/env python
"""
简单的模型转换命令行工具，使用cn_clip_model_converter.py中的转换函数
"""

import argparse
import os
from cn_clip_model_converter import convert_muge_model_to_chinese_clip

def parse_args():
    parser = argparse.ArgumentParser(description="转换MUGE模型为Chinese-CLIP可用格式（简化版）")
    parser.add_argument(
        "--input_model", 
        type=str, 
        required=True,
        help="输入模型路径"
    )
    parser.add_argument(
        "--output_model", 
        type=str, 
        required=True,
        help="输出模型路径"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    
    print(f"正在转换模型 {args.input_model} -> {args.output_model}")
    
    # 转换模型
    convert_muge_model_to_chinese_clip(args.input_model, args.output_model)
    
    print("模型转换完成!")

if __name__ == "__main__":
    main() 