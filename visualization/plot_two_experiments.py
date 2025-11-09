#!/usr/bin/env python3
"""
绘制两个新实验的loss曲线对比图
- 实验1: BP和ZO使用不同的数据子集（split_bp_zo）
- 实验2: ZO的64个方向共享128条数据（shared_zo_data）
"""

import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def find_csv_files():
    """查找两个实验的CSV文件"""
    csv_files = {}
    
    # 查找split实验
    split_pattern = "logs/*split*/*.csv"
    split_files = glob.glob(split_pattern)
    if split_files:
        latest_split = max(split_files, key=os.path.getmtime)
        csv_files['exp1'] = latest_split
        print(f"找到实验1 (split): {latest_split}")
    else:
        print(f"未找到split实验的CSV文件 (pattern: {split_pattern})")
    
    # 查找shared实验
    shared_pattern = "logs/*shared*/*.csv"
    shared_files = glob.glob(shared_pattern)
    if shared_files:
        latest_shared = max(shared_files, key=os.path.getmtime)
        csv_files['exp2'] = latest_shared
        print(f"找到实验2 (shared): {latest_shared}")
    else:
        print(f"未找到shared实验的CSV文件 (pattern: {shared_pattern})")
    
    return csv_files

def load_csv_data(csv_file):
    """加载CSV数据"""
    try:
        df = pd.read_csv(csv_file)
        # CSV列名: timestamp, epoch, step, mode, scope, q, lr, batch_size, optimizer, bp_interval, loss, grad_norm, eval_loss
        if 'step' in df.columns and 'loss' in df.columns:
            return df
        else:
            print(f"警告: {csv_file} 缺少必要的列")
            print(f"  现有列: {df.columns.tolist()}")
            return None
    except Exception as e:
        print(f"读取 {csv_file} 失败: {e}")
        return None

def plot_loss_curves(csv_files, output_file):
    """绘制loss曲线对比图"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 实验标签和颜色
    exp_configs = {
        'exp1': {
            'label': 'Exp1: BP and ZO use different data subsets (split)',
            'color': '#1f77b4',  # 蓝色
        },
        'exp2': {
            'label': 'Exp2: ZO 64 directions share 128 samples (shared)',
            'color': '#ff7f0e',  # 橙色
        },
    }
    
    for exp_id in ['exp1', 'exp2']:
        if exp_id not in csv_files:
            continue
        
        csv_file = csv_files[exp_id]
        df = load_csv_data(csv_file)
        
        if df is None:
            continue
        
        config = exp_configs[exp_id]
        
        # 绘制训练loss
        if 'step' in df.columns and 'loss' in df.columns:
            steps = df['step'].values
            losses = df['loss'].values
            
            # 过滤掉无效值
            valid_mask = pd.notna(steps) & pd.notna(losses)
            steps = steps[valid_mask]
            losses = losses[valid_mask]
            
            if len(steps) > 0:
                # 每隔100步取一次值
                step_mask = (steps % 100 == 0) | (steps == steps[-1])  # 包含最后一个点
                sampled_steps = steps[step_mask]
                sampled_losses = losses[step_mask]
                
                if len(sampled_steps) > 0:
                    label = f"{config['label']} (train)"
                    ax.plot(sampled_steps, sampled_losses, label=label, color=config['color'], 
                           linestyle='-', linewidth=2, alpha=0.7)
        
        # 绘制评估loss（如果有）
        if 'eval_loss' in df.columns and 'step' in df.columns:
            # 只取有评估loss的步骤（评估loss本身就是每100步记录一次）
            eval_mask = df['eval_loss'].notna() & (df['eval_loss'] != '')
            if eval_mask.any():
                eval_steps = df.loc[eval_mask, 'step'].values
                eval_losses = df.loc[eval_mask, 'eval_loss'].values
                
                # 转换为数值类型
                try:
                    eval_losses = pd.to_numeric(eval_losses, errors='coerce')
                    valid_eval_mask = pd.notna(eval_losses)
                    eval_steps = eval_steps[valid_eval_mask]
                    eval_losses = eval_losses[valid_eval_mask]
                    
                    if len(eval_steps) > 0:
                        # 评估loss已经是每100步记录一次，所以不需要再采样
                        label = f"{config['label']} (eval)"
                        ax.plot(eval_steps, eval_losses, label=label, 
                               color=config['color'], linestyle='--', 
                               linewidth=2, alpha=0.8, marker='o', markersize=4)
                except Exception as e:
                    print(f"处理 {exp_id} 的评估loss时出错: {e}")
    
    ax.set_xlabel('Step', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.set_title('Loss Curves: BP/ZO Data Usage Comparison', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"已保存图表到: {output_file}")
    plt.close()

def main():
    print("=" * 60)
    print("开始绘制两个实验的loss曲线对比图")
    print("=" * 60)
    
    # 查找CSV文件
    csv_files = find_csv_files()
    
    if not csv_files:
        print("错误: 未找到任何CSV文件")
        return
    
    if len(csv_files) < 2:
        print(f"警告: 只找到 {len(csv_files)} 个实验的CSV文件，将只绘制找到的实验")
    
    # 创建输出目录
    output_dir = Path('plots')
    output_dir.mkdir(exist_ok=True)
    
    # 绘制对比图
    print("\n绘制两个实验的对比图...")
    plot_loss_curves(
        csv_files,
        output_dir / 'loss_curves_two_experiments.png'
    )
    
    print("\n" + "=" * 60)
    print("图表绘制完成！")
    print("=" * 60)

if __name__ == '__main__':
    main()

