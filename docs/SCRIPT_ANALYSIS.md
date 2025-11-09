# 脚本关系分析文档

本文档详细分析了 `zo_withbp` 目录中所有脚本的关系和依赖。

## 📁 目录结构

```
zo_withbp/
├── 核心模块
│   ├── model.py              # 模型定义和配置
│   ├── data.py               # 数据集加载和配置
│   └── reproduce_zo_paper_*.py  # 训练脚本（多个版本）
│
├── 数据管理脚本
│   ├── download_datasets.py      # 下载数据集
│   ├── merge_datasets.py         # 合并数据集
│   ├── check_dataset_size.py      # 检查数据集大小
│   └── check_data_distribution.py # 检查数据分布
│
├── 可视化脚本
│   ├── plot_all_results.py        # 综合分析绘图
│   ├── plot_loss_curves.py       # 损失曲线绘图
│   ├── plot_two_experiments.py    # 双实验对比绘图
│   └── quick_plot.py              # 快速绘图工具
│
├── 测试脚本
│   ├── test_setup.py              # 环境测试
│   ├── test_training.py            # 训练测试
│   ├── test_quick.py               # 快速测试
│   ├── test_merged_dataset.py      # 数据集测试
│   └── test_zo_vs_fo.py           # ZO vs FO 对比测试
│
├── 实验运行脚本（Shell）
│   ├── run_two_experiments.sh      # 运行两个对比实验
│   ├── run_experiments.sh          # 标准实验
│   ├── run_zo_batch_size_experiments.sh  # ZO batch size 实验
│   ├── parallel_sweep.sh           # 并行参数扫描
│   └── run_and_plot.sh             # 运行并绘图
│
├── 特殊用途脚本
│   ├── flwr_server.py              # Flower 联邦学习服务器
│   └── zo_sst_finetune.py          # SST-2 微调
│
└── 数据目录
    ├── tokenizer/                  # Tokenizer 文件
    └── datasets_subset/            # 本地数据集
```

## 🔗 脚本依赖关系

### 核心模块依赖图

```
reproduce_zo_paper_1106.py (主训练脚本)
    ├── model.py                    # 导入: create_model()
    ├── data.py                     # 导入: get_dataloader()
    ├── torch                       # PyTorch 深度学习框架
    ├── transformers                # HuggingFace Transformers
    ├── datasets                    # HuggingFace Datasets
    └── matplotlib                  # 绘图库

model.py (模型定义)
    └── transformers                # 导入: GPT2Config, GPT2LMHeadModel

data.py (数据加载)
    ├── datasets                    # 导入: load_dataset, load_from_disk
    ├── torch                       # 导入: DataLoader
    └── tqdm                        # 进度条
```

### 训练脚本版本关系

```
reproduce_zo_paper.py (原始版本)
    ↓
reproduce_zo_paper_withbp.py (添加BP支持)
    ↓
reproduce_zo_paper_new.py (新实现)
    ↓
reproduce_zo_paper_1105.py (添加评估功能)
    ↓
reproduce_zo_paper_1106.py (最新版本，功能最全)
    ├── 支持数据集分割
    ├── 支持共享ZO数据
    ├── 支持评估
    └── 支持多种优化器
```

### 可视化脚本依赖

```
plot_all_results.py
    ├── pandas                      # 数据处理
    ├── matplotlib                  # 基础绘图
    ├── seaborn                     # 高级可视化
    └── numpy                       # 数值计算

plot_loss_curves.py
    ├── pandas
    └── matplotlib

plot_two_experiments.py
    ├── pandas
    └── matplotlib

quick_plot.py
    ├── pandas
    └── matplotlib
```

### 特殊脚本依赖

```
flwr_server.py (联邦学习)
    ├── flwr                        # Flower 框架
    ├── numpy
    └── model.py                    # 导入: get_model()

zo_sst_finetune.py (微调)
    ├── transformers                # AutoTokenizer, AutoModelForSequenceClassification
    ├── datasets                    # load_dataset
    └── torch
```

## 📦 模块功能说明

### 1. 核心训练模块

**model.py**
- 功能：定义GPT-2模型配置和创建函数
- 导出：`create_model()`, `get_model_info()`, `list_available_models()`
- 被导入：所有训练脚本

**data.py**
- 功能：数据集配置和加载
- 导出：`get_dataloader()`, `get_dataset_info()`, `list_available_datasets()`
- 被导入：所有训练脚本、测试脚本

**reproduce_zo_paper_1106.py**
- 功能：主训练脚本（最新版本）
- 特性：
  - 支持 FO/ZO/Instruct 三种模式
  - 支持数据集分割和共享
  - 支持评估
  - 支持多种优化器（SGD/Adam/MuDaMW）
- 被调用：Shell脚本、测试脚本

### 2. 数据管理模块

**download_datasets.py**
- 功能：从 HuggingFace 下载数据集
- 依赖：`datasets`

**merge_datasets.py**
- 功能：合并多个本地数据集
- 依赖：`datasets`, `tqdm`
- 输出：`datasets_subset/dclm_pubmedqa_merged/`

**check_dataset_size.py**
- 功能：检查数据集大小
- 依赖：`datasets`

**check_data_distribution.py**
- 功能：验证数据集分布
- 依赖：`datasets`

### 3. 可视化模块

所有绘图脚本都依赖：
- `pandas`：读取CSV日志
- `matplotlib`：绘图
- `seaborn`（部分）：高级可视化

### 4. 测试模块

测试脚本通常导入训练脚本的 `train()` 函数进行测试。

## 🔄 数据流

```
数据集 (HuggingFace/本地)
    ↓
data.py::get_dataloader()
    ↓
reproduce_zo_paper_1106.py::train()
    ↓
模型训练 (model.py::create_model())
    ↓
CSV日志文件
    ↓
可视化脚本 (plot_*.py)
    ↓
图表输出
```

## 🚀 典型使用流程

### 1. 数据准备流程
```bash
# 下载数据集
python download_datasets.py

# 合并数据集（可选）
python merge_datasets.py

# 检查数据集
python check_dataset_size.py
python check_data_distribution.py
```

### 2. 训练流程
```bash
# 直接运行训练
python reproduce_zo_paper_1106.py --mode ZO --scope full ...

# 或使用Shell脚本批量运行
./run_two_experiments.sh
./run_experiments.sh
```

### 3. 结果分析流程
```bash
# 快速绘图
python quick_plot.py

# 详细分析
python plot_all_results.py
python plot_loss_curves.py
python plot_two_experiments.py
```

## 📝 注意事项

1. **路径依赖**：
   - `data.py` 中的本地数据集路径相对于脚本目录
   - `reproduce_zo_paper_1106.py` 中的 tokenizer 路径相对于脚本目录
   - 所有路径已修改为基于脚本目录的绝对路径

2. **版本选择**：
   - 推荐使用 `reproduce_zo_paper_1106.py`（最新版本）
   - 其他版本保留用于兼容性

3. **依赖安装**：
   - 所有依赖在 `requirements.txt` 中列出
   - 某些脚本需要特定版本（如 CUDA 版本的 PyTorch）

4. **数据目录**：
   - `tokenizer/`：必须存在，包含 tokenizer 文件
   - `datasets_subset/`：本地数据集目录

## 🔍 脚本调用关系示例

### 示例1：运行训练实验
```bash
# Shell脚本调用Python训练脚本
./run_two_experiments.sh
    → python reproduce_zo_paper_1106.py ...
        → from model import create_model
        → from data import get_dataloader
```

### 示例2：分析结果
```bash
# 绘图脚本读取训练生成的CSV
python plot_two_experiments.py
    → 读取 logs/*.csv
    → 使用 pandas 处理数据
    → 使用 matplotlib 绘图
```

### 示例3：测试流程
```bash
# 测试脚本导入训练函数
python test_training.py
    → from reproduce_zo_paper_1105 import train
    → 调用 train() 进行测试
```

