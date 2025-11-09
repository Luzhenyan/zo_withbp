#!/usr/bin/env python3
"""
小规模测试脚本 - 验证数据加载和训练流程
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer
from core.data import get_dataloader, get_dataset_info
from core.model import create_model

print("=" * 80)
print("开始测试数据加载和模型创建")
print("=" * 80)

# 1. 测试tokenizer加载
print("\n[1] 测试 Tokenizer 加载...")
try:
    tokenizer_path = "/data/cdq/.conda/envs/speechbrain/lib/python3.9/site-packages/whisper/assets/gpt2"
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        print(f"✅ 成功从本地加载tokenizer: {tokenizer_path}")
    except Exception as e:
        print(f"⚠️  无法从本地路径加载: {e}")
        print("尝试从在线加载...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        print("✅ 成功从在线加载tokenizer")
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        print("✅ 添加了pad_token")
except Exception as e:
    print(f"❌ Tokenizer加载失败: {e}")
    exit(1)

# 2. 测试数据集配置信息
print("\n[2] 测试数据集配置信息...")
for dataset_name in ['dclm-local', 'pubmedqa-local', 'cosmopedia']:
    try:
        info = get_dataset_info(dataset_name)
        print(f"✅ {dataset_name}: {info['description']}")
    except Exception as e:
        print(f"❌ {dataset_name} 配置错误: {e}")

# 3. 测试dclm-local数据加载（小样本）
print("\n[3] 测试 dclm-local 数据加载（限制10个样本）...")
try:
    dclm_loader = get_dataloader(
        tokenizer=tokenizer,
        dataset_name='dclm-local',
        batch_size=2,
        block_size=128,
        max_samples=10,  # 只加载10个样本用于测试
    )
    print(f"✅ dclm-local数据加载成功")
    
    # 测试获取一个batch
    batch = next(iter(dclm_loader))
    print(f"✅ Batch形状: {batch.shape}, dtype: {batch.dtype}")
    print(f"   样本数量: {len(batch)}")
except Exception as e:
    print(f"❌ dclm-local数据加载失败: {e}")
    import traceback
    traceback.print_exc()

# 4. 测试pubmedqa-local数据加载（小样本）
print("\n[4] 测试 pubmedqa-local 数据加载（限制10个样本）...")
try:
    pubmedqa_loader = get_dataloader(
        tokenizer=tokenizer,
        dataset_name='pubmedqa-local',
        batch_size=2,
        block_size=128,
        max_samples=10,  # 只加载10个样本用于测试
    )
    print(f"✅ pubmedqa-local数据加载成功")
    
    # 测试获取一个batch
    batch = next(iter(pubmedqa_loader))
    print(f"✅ Batch形状: {batch.shape}, dtype: {batch.dtype}")
    print(f"   样本数量: {len(batch)}")
except Exception as e:
    print(f"❌ pubmedqa-local数据加载失败: {e}")
    import traceback
    traceback.print_exc()

# 5. 测试模型创建
print("\n[5] 测试模型创建（20M用于快速测试）...")
try:
    model = create_model(model_size='20M', vocab_size=len(tokenizer))
    print(f"✅ 模型创建成功")
    print(f"   参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
except Exception as e:
    print(f"❌ 模型创建失败: {e}")
    import traceback
    traceback.print_exc()

# 6. 测试前向传播
print("\n[6] 测试模型前向传播...")
try:
    if 'dclm_loader' in locals():
        test_batch = next(iter(dclm_loader))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   使用设备: {device}")
        
        test_inputs = test_batch[:2].to(device)  # 只用前2个样本
        model = model.to(device)
        
        with torch.no_grad():
            outputs = model(test_inputs)
            print(f"✅ 前向传播成功")
            print(f"   Logits形状: {outputs.logits.shape}")
            print(f"   损失计算: {torch.nn.functional.cross_entropy(outputs.logits[:, :-1].reshape(-1, outputs.logits.size(-1)), test_inputs[:, 1:].reshape(-1)).item():.4f}")
except Exception as e:
    print(f"❌ 前向传播失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("测试完成！")
print("=" * 80)

