#!/usr/bin/env python3
"""合并 DCLM 和 PubMedQA 数据集"""

from datasets import load_from_disk, Dataset, concatenate_datasets
from tqdm import tqdm
import os

print("=" * 80)
print("合并数据集：DCLM + PubMedQA")
print("=" * 80)

# 1. 加载两个数据集
print("\n[1] 加载数据集...")
dclm_path = './datasets_subset/dclm'
pubmedqa_path = './datasets_subset/pubmedqa'

dclm = load_from_disk(dclm_path)
pubmedqa = load_from_disk(pubmedqa_path)

print(f"DCLM: {len(dclm):,} 条")
print(f"PubMedQA: {len(pubmedqa):,} 条")

# 2. 统一字段格式 - 将两个数据集都转换为只有 'text' 字段
print("\n[2] 统一字段格式...")

def process_dclm(example):
    """处理 DCLM 数据：直接使用 text 字段"""
    return {'text': example['text']}

def process_pubmedqa(example):
    """处理 PubMedQA 数据：拼接 question + long_answer + final_decision"""
    text_parts = []
    
    # 添加 question
    if example.get('question'):
        text_parts.append(str(example['question']))
    
    # 添加 long_answer
    if example.get('long_answer'):
        text_parts.append(str(example['long_answer']))
    
    # 添加 final_decision
    if example.get('final_decision'):
        text_parts.append(f"Answer: {example['final_decision']}")
    
    # 拼接所有部分
    text = " ".join(text_parts)
    return {'text': text}

print("处理 DCLM 数据...")
dclm_processed = dclm.map(process_dclm, remove_columns=[col for col in dclm.column_names if col != 'text'])

print("处理 PubMedQA 数据...")
pubmedqa_processed = pubmedqa.map(process_pubmedqa, remove_columns=[col for col in pubmedqa.column_names if col != 'text'])

# 3. 合并数据集
print("\n[3] 合并数据集...")
merged_dataset = concatenate_datasets([dclm_processed, pubmedqa_processed])

print(f"合并后总数据量: {len(merged_dataset):,} 条")
print(f"  - DCLM: {len(dclm_processed):,} 条")
print(f"  - PubMedQA: {len(pubmedqa_processed):,} 条")

# 4. 打乱数据
print("\n[4] 打乱数据顺序...")
merged_dataset = merged_dataset.shuffle(seed=42)

# 5. 保存数据集
print("\n[5] 保存合并后的数据集...")
output_path = './datasets_subset/dclm_pubmedqa_merged'
merged_dataset.save_to_disk(output_path)

print(f"数据集已保存到: {output_path}")

# 6. 验证保存的数据集
print("\n[6] 验证保存的数据集...")
verify_dataset = load_from_disk(output_path)
print(f"验证: 加载了 {len(verify_dataset):,} 条数据")
print(f"特征: {list(verify_dataset.features.keys())}")

# 显示几个示例
print("\n[7] 数据示例:")
print("\n示例 1 (来自 DCLM):")
print(verify_dataset[0]['text'][:200] + "...")

print("\n示例 2 (来自 PubMedQA):")
print(verify_dataset[len(dclm)]['text'][:200] + "...")

# 检查文件大小
if os.path.isdir(output_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(output_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    print(f"\n数据集文件大小: {total_size / 1024 / 1024:.2f} MB")

print("\n" + "=" * 80)
print("数据集合并完成！")
print("=" * 80)


