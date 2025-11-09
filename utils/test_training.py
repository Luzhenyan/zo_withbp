#!/usr/bin/env python3
"""
小规模训练测试 - 验证训练流程和评估功能
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.reproduce_zo_paper_1105 import train
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_training")

print("=" * 80)
print("开始小规模训练测试")
print("=" * 80)

# 测试参数
test_params = {
    'mode': 'FO',
    'scope': 'full',
    'q': None,  # FO模式不需要
    'lr': 0.001,
    'epochs': 1,  # 只训练1个epoch
    'batch_size': 2,
    'device': 'cuda' if __import__('torch').cuda.is_available() else 'cpu',
    'plot_file': Path('test_results/test_training_loss.png'),
    'csv_file': 'test_results/test_training.csv',
    'log_interval': 5,  # 每5步记录一次
    'optimizer_type': 'mudamw',
    'bp_interval': None,
    'queries_use_different_data': False,
    'model_size': '20M',  # 使用小模型
    'dataset_name': 'dclm-local',
    'max_samples': 20,  # 只用20个样本
    'checkpoint_dir': None,  # 不保存checkpoint
    'logger': logger,
    'run_name': 'test_training',
    'bp_dataset_name': None,
    'bp_max_samples': None,
    'eval_dataset_name': 'cosmopedia-100k',  # 测试评估功能（使用100k版本，不需要config）
    'eval_interval': 10,  # 每10步评估一次
}

# 创建结果目录
test_results_dir = Path('test_results')
test_results_dir.mkdir(exist_ok=True)

print("\n测试配置:")
for key, value in test_params.items():
    print(f"  {key}: {value}")

print("\n开始训练...")
try:
    train(**test_params)
    print("\n✅ 训练测试成功完成！")
except Exception as e:
    print(f"\n❌ 训练测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=" * 80)
print("测试完成！检查 test_results/ 目录查看结果")
print("=" * 80)

