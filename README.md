# Zero-Order Optimization (ZO) Training Project

零阶优化训练项目 - 使用零阶优化方法训练GPT-2语言模型

## 📁 目录结构

```
zo_withbp/
├── core/                    # 核心模块和训练脚本
│   ├── __init__.py
│   ├── model.py            # 模型定义和配置
│   ├── data.py             # 数据集加载和配置
│   ├── reproduce_zo_paper.py              # 原始版本
│   ├── reproduce_zo_paper_withbp.py      # 添加BP支持
│   ├── reproduce_zo_paper_new.py         # 新实现
│   ├── reproduce_zo_paper_1105.py        # 添加评估功能
│   └── reproduce_zo_paper_1106.py        # 最新版本（推荐）
│
├── utils/                   # 工具脚本
│   ├── download_datasets.py       # 下载数据集
│   ├── merge_datasets.py          # 合并数据集
│   ├── check_dataset_size.py      # 检查数据集大小
│   ├── check_data_distribution.py # 检查数据分布
│   └── test_*.py                  # 测试脚本
│
├── visualization/           # 可视化脚本
│   ├── plot_all_results.py        # 综合分析绘图
│   ├── plot_loss_curves.py        # 损失曲线绘图
│   ├── plot_two_experiments.py    # 双实验对比绘图
│   └── quick_plot.py              # 快速绘图工具
│
├── experiments/             # 实验运行脚本（Shell）
│   ├── run_two_experiments.sh           # 运行两个对比实验
│   ├── run_experiments.sh               # 标准实验
│   ├── run_zo_batch_size_experiments.sh  # ZO batch size 实验
│   ├── parallel_sweep.sh                # 并行参数扫描
│   └── run_and_plot.sh                  # 运行并绘图
│
├── special/                 # 特殊用途脚本
│   ├── flwr_server.py       # Flower 联邦学习服务器
│   └── zo_sst_finetune.py   # SST-2 微调
│
├── docs/                    # 文档
│   ├── README_SCRIPTS.md    # 脚本详细文档
│   ├── README_parallel.md   # 并行实验文档
│   ├── README_plotting.md   # 绘图文档
│   ├── README_sweep.md      # 参数扫描文档
│   ├── SCRIPT_ANALYSIS.md   # 脚本关系分析
│   └── requirements.txt     # Python依赖包
│
├── tokenizer/               # Tokenizer 文件
└── datasets_subset/         # 本地数据集
```

## 🚀 快速开始

### 1. 安装Conda环境

**方法一：使用安装脚本（推荐）**
```bash
./setup_conda_env.sh
```

**方法二：手动安装**
```bash
# 创建环境
conda env create -f environment.yml

# 激活环境
conda activate zo_withbp

# 如果使用GPU，需要安装CUDA版本的PyTorch
# CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1 (最低版本2.1.0):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# 或指定版本（CUDA 12.1推荐2.1.2）:
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
```

详细安装说明请查看 [docs/INSTALL.md](docs/INSTALL.md)

### 2. 运行训练

```bash
# 使用最新版本的训练脚本
python core/reproduce_zo_paper_1106.py \
    --mode ZO \
    --scope full \
    --query_budget_q 64 \
    --learning_rate 0.001 \
    --batch_size 2 \
    --dataset cosmopedia-100k
```

### 3. 运行实验

```bash
# 运行两个对比实验
cd experiments
./run_two_experiments.sh
```

### 4. 可视化结果

```bash
# 快速绘图
python visualization/quick_plot.py

# 详细分析
python visualization/plot_all_results.py
```

## 📖 文档

详细文档请查看 `docs/` 目录：

- **README_SCRIPTS.md**: 所有脚本的详细说明
- **SCRIPT_ANALYSIS.md**: 脚本关系分析和依赖说明
- **requirements.txt**: Python依赖包列表

## 🔧 核心模块

### core/model.py
- 定义GPT-2模型配置（20M, 200M, 500M, 1B）
- 模型规模：
  - 20M: ~17M 参数
  - 200M: ~200M 参数
  - 500M: ~500M 参数
  - 1B: ~1066M 参数（真正接近1B）
- 提供 `create_model()` 函数创建模型

### core/data.py
- 数据集配置和加载
- 支持多种数据集（Cosmopedia, WikiText, 本地数据集等）
- 本地数据集：dclm-local, pubmedqa-local, dclm-pubmedqa-merged
- 提供 `get_dataloader()` 函数加载数据
- 自动限制生成的数据块数量以加快测试

## 📝 注意事项

1. **路径说明**：
   - 所有脚本使用相对于项目根目录的路径
   - 核心模块和训练脚本都在 `core/` 目录

2. **运行脚本**：
   - 从项目根目录运行Python脚本
   - Shell脚本在 `experiments/` 目录，会自动切换到项目根目录

3. **数据目录**：
   - `tokenizer/`: Tokenizer文件（必须存在）
   - `datasets_subset/`: 本地数据集目录

4. **推荐使用**：
   - 训练脚本：`core/reproduce_zo_paper_1106.py`（最新版本，功能最全）
   - 实验脚本：`experiments/run_merged_dataset_experiments.sh`（混合数据集对比实验）
   - 测试脚本：`experiments/test_1b_model.sh`（1B模型测试）

## 🔗 相关链接

- 项目文档：`docs/README_SCRIPTS.md`
- 脚本分析：`docs/SCRIPT_ANALYSIS.md`
- 依赖列表：`docs/requirements.txt`
- GitHub 仓库：https://github.com/Luzhenyan/zo_withbp

## 📊 实验脚本

### 混合数据集对比实验
```bash
./experiments/run_merged_dataset_experiments.sh
```
三个实验对比不同的数据使用策略：
1. BP和ZO数据分割，ZO各方向独立数据
2. BP和ZO数据分割，ZO各方向共享数据
3. 纯BP训练（基线）

### 1B模型测试
```bash
./experiments/test_1b_model.sh
```
测试1B模型在三种模式下是否能正常运行：
1. BP only (FO模式)
2. ZO only
3. BP+ZO (Instruct模式)

## 🔄 最近更新

- **2024-11-10**: 
  - 修复1B模型CUDA OOM问题（将随机排列生成移到CPU）
  - 增大1B模型参数到1066M（更接近真正的1B）
  - 修复数据加载：限制生成的blocks数量
  - 创建混合数据集实验脚本
  - 完善conda环境配置

