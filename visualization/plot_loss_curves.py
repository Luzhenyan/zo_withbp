#!/usr/bin/env python3
"""
绘制loss曲线图
- exp1-3画到一张图
- exp4-6画到另一张图
- 包含训练loss和评估loss（如果有）
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
    """查找所有实验的CSV文件"""
    csv_files = {}
    
    # 实验名称映射
    exp_names = {
        'exp1': 'FO_dclm_train_cosmopedia_eval',
        'exp2': 'Instruct_cosmopedia_train_dclm_bp',
        'exp3': 'ZO_cosmopedia_train',
        'exp4': 'FO_dclm_train_pubmedqa_eval',
        'exp5': 'Instruct_pubmedqa_train_dclm_bp',
        'exp6': 'ZO_pubmedqa_train',
    }
    
    # 在logs目录下查找最新的CSV文件
    for exp_id, name_pattern in exp_names.items():
        pattern = f"logs/{name_pattern}*/*.csv"
        files = glob.glob(pattern)
        if files:
            # 选择最新的文件
            latest_file = max(files, key=os.path.getmtime)
            csv_files[exp_id] = latest_file
            print(f"找到 {exp_id}: {latest_file}")
        else:
            print(f"未找到 {exp_id} 的CSV文件 (pattern: {pattern})")
    
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

def plot_loss_curves(exp_ids, csv_files, output_file, title):
    """绘制loss曲线"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 实验标签
    exp_labels = {
        'exp1': 'FO (dclm train, cosmopedia eval)',
        'exp2': 'Instruct (cosmopedia train, dclm bp)',
        'exp3': 'ZO (cosmopedia train)',
        'exp4': 'FO (dclm train, pubmedqa eval)',
        'exp5': 'Instruct (pubmedqa train, dclm bp)',
        'exp6': 'ZO (pubmedqa train)',
    }
    
    colors = {
        'exp1': '#1f77b4',  # 蓝色
        'exp2': '#ff7f0e',  # 橙色
        'exp3': '#2ca02c',  # 绿色
        'exp4': '#1f77b4',  # 蓝色
        'exp5': '#ff7f0e',  # 橙色
        'exp6': '#2ca02c',  # 绿色
    }
    
    for exp_id in exp_ids:
        if exp_id not in csv_files:
            continue
        
        csv_file = csv_files[exp_id]
        df = load_csv_data(csv_file)
        
        if df is None:
            continue
        
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
                    label = f"{exp_labels[exp_id]} (train)"
                    ax.plot(sampled_steps, sampled_losses, label=label, color=colors[exp_id], 
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
                        label = f"{exp_labels[exp_id]} (eval)"
                        ax.plot(eval_steps, eval_losses, label=label, 
                               color=colors[exp_id], linestyle='--', 
                               linewidth=2, alpha=0.8, marker='o', markersize=4)
                except Exception as e:
                    print(f"处理 {exp_id} 的评估loss时出错: {e}")
    
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"已保存图表到: {output_file}")
    plt.close()

def main():
    print("=" * 60)
    print("开始绘制loss曲线")
    print("=" * 60)
    
    # 查找所有CSV文件
    csv_files = find_csv_files()
    
    if not csv_files:
        print("错误: 未找到任何CSV文件")
        return
    
    # 创建输出目录
    output_dir = Path('plots')
    output_dir.mkdir(exist_ok=True)
    
    # 绘制exp1-3
    print("\n绘制 exp1-3 (dclm + cosmopedia)...")
    plot_loss_curves(
        ['exp1', 'exp2', 'exp3'],
        csv_files,
        output_dir / 'loss_curves_exp1-3.png',
        'Loss Curves: dclm + cosmopedia experiments'
    )
    
    # 绘制exp4-6
    print("\n绘制 exp4-6 (dclm + pubmedqa)...")
    plot_loss_curves(
        ['exp4', 'exp5', 'exp6'],
        csv_files,
        output_dir / 'loss_curves_exp4-6.png',
        'Loss Curves: dclm + pubmedqa experiments'
    )
    
    print("\n" + "=" * 60)
    print("所有图表绘制完成！")
    print("=" * 60)

if __name__ == '__main__':
    main()

