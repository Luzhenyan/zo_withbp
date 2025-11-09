# 环境安装指南

本文档说明如何为ZO训练项目设置conda环境。

## 快速安装

### 方法一：使用安装脚本（推荐）

```bash
cd /home/polyu/pcllzy/zo_withbp
./setup_conda_env.sh
```

脚本会自动：
- 检测是否有GPU
- 选择合适的PyTorch版本
- 创建conda环境
- 安装所有依赖

### 方法二：手动安装

#### 1. 创建conda环境

```bash
# CPU版本（简单）
conda env create -f environment.yml

# 或使用CUDA版本（需要GPU）
conda env create -f environment_cuda.yml
```

#### 2. 激活环境

```bash
conda activate zo_withbp
```

#### 3. 安装CUDA版本的PyTorch（如果使用GPU）

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1（自动安装最新版本，最低2.1.0）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 或指定具体版本（CUDA 12.1推荐2.1.2，稳定版本）
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
```

#### 4. 验证安装

```bash
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers版本: {transformers.__version__}')"
```

## 环境要求

### Python版本
- Python 3.9（推荐）

### 主要依赖
- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- Datasets >= 2.12.0
- NumPy >= 1.24.0
- Pandas >= 2.0.0
- Matplotlib >= 3.7.0
- Seaborn >= 0.12.0
- tqdm >= 4.65.0
- flwr >= 1.4.0（可选，仅联邦学习需要）

## CUDA版本选择

### 检查CUDA版本

```bash
nvidia-smi
```

查看 "CUDA Version" 行。

### PyTorch CUDA版本对应

| CUDA版本 | PyTorch安装命令 |
|---------|----------------|
| 11.8 | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` |
| 12.1 | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121` |
| 12.1 (指定版本) | `pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121` |
| CPU | `pip install torch torchvision torchaudio` |

## 常见问题

### 1. 环境已存在

如果环境已存在，可以：

```bash
# 删除并重新创建
conda env remove -n zo_withbp
conda env create -f environment.yml
```

### 2. 安装失败

如果安装失败，尝试：

```bash
# 更新conda
conda update conda

# 清理缓存
conda clean --all

# 重新安装
conda env create -f environment.yml --force
```

### 3. CUDA不可用

如果安装CUDA版本后仍显示CUDA不可用：

1. 检查CUDA驱动：
   ```bash
   nvidia-smi
   ```

2. 检查PyTorch CUDA版本：
   ```bash
   python -c "import torch; print(torch.version.cuda)"
   ```

3. 确保驱动版本 >= CUDA版本

### 4. 依赖冲突

如果遇到依赖冲突：

```bash
# 使用pip安装（可能更灵活）
conda activate zo_withbp
pip install -r docs/requirements.txt
```

## 环境管理

### 查看环境列表

```bash
conda env list
```

### 激活环境

```bash
conda activate zo_withbp
```

### 退出环境

```bash
conda deactivate
```

### 删除环境

```bash
conda env remove -n zo_withbp
```

### 导出环境（备份）

```bash
conda env export > environment_backup.yml
```

## 最小安装（仅核心功能）

如果只需要核心训练功能，可以只安装：

```bash
conda create -n zo_withbp python=3.9
conda activate zo_withbp
pip install torch transformers datasets tqdm matplotlib
```

## 验证安装

运行测试脚本验证环境：

```bash
conda activate zo_withbp
python utils/test_setup.py
```

如果所有测试通过，说明环境配置正确。

