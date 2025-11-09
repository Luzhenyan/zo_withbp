#!/usr/bin/env python3
"""
快速测试 - 只运行几步训练和评估
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.reproduce_zo_paper_1105 import train
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_quick")

print("=" * 80)
print("快速测试：只运行10步训练")
print("=" * 80)

# 快速测试参数 - 使用更小的数据集和更少的步骤
test_params = {
    'mode': 'FO',
    'scope': 'full',
    'q': None,
    'lr': 0.001,
    'epochs': 1,
    'batch_size': 2,
    'device': 'cuda' if __import__('torch').cuda.is_available() else 'cpu',
    'plot_file': Path('test_results/quick_test_loss.png'),
    'csv_file': 'test_results/quick_test.csv',
    'log_interval': 2,  # 每2步记录
    'optimizer_type': 'mudamw',
    'bp_interval': None,
    'queries_use_different_data': False,
    'model_size': '20M',
    'dataset_name': 'dclm-local',
    'max_samples': 5,  # 只用5个样本，这样只有很少的batches
    'checkpoint_dir': None,
    'logger': logger,
    'run_name': 'quick_test',
    'bp_dataset_name': None,
    'bp_max_samples': None,
    'eval_dataset_name': 'pubmedqa-local',  # 使用本地数据集作为评估集
    'eval_interval': 5,  # 每5步评估
}

Path('test_results').mkdir(exist_ok=True)

print("\n测试配置:")
for key, value in test_params.items():
    print(f"  {key}: {value}")

print("\n开始训练（将很快完成）...")
try:
    train(**test_params)
    print("\n✅ 快速测试成功完成！")
    print("检查 test_results/ 目录查看结果")
except KeyboardInterrupt:
    print("\n⚠️  测试被中断（这是正常的，我们只测试基本功能）")
except Exception as e:
    print(f"\n❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()

print("=" * 80)

