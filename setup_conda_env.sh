#!/bin/bash
# Conda环境安装脚本
# 用于创建和配置zo_withbp项目的conda环境

set -e

ENV_NAME="zo_withbp"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================="
echo "ZO Training Project - Conda环境安装"
echo "=========================================="
echo ""

# 检查conda是否安装
if ! command -v conda &> /dev/null; then
    echo "❌ 错误: 未找到conda，请先安装Anaconda或Miniconda"
    exit 1
fi

echo "✅ 找到conda: $(conda --version)"
echo ""

# 检查CUDA是否可用
if command -v nvidia-smi &> /dev/null; then
    echo "检测到NVIDIA GPU:"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2 || echo "unknown")
    echo "CUDA版本: $CUDA_VERSION"
    echo ""
    
    USE_CUDA=true
else
    echo "⚠️  未检测到NVIDIA GPU，将安装CPU版本的PyTorch"
    USE_CUDA=false
    echo ""
fi

# 选择环境文件
if [ "$USE_CUDA" = true ]; then
    echo "选择安装方式:"
    echo "  1) 使用CPU版本PyTorch（简单，但速度慢）"
    echo "  2) 使用CUDA版本PyTorch（需要手动指定CUDA版本）"
    read -p "请选择 [1/2] (默认: 1): " choice
    choice=${choice:-1}
    
    if [ "$choice" = "2" ]; then
        ENV_FILE="environment_cuda.yml"
        echo "使用CUDA环境配置"
    else
        ENV_FILE="environment.yml"
        echo "使用CPU环境配置"
    fi
else
    ENV_FILE="environment.yml"
    echo "使用CPU环境配置"
fi

echo ""
echo "环境名称: $ENV_NAME"
echo "环境文件: $ENV_FILE"
echo ""

# 检查环境是否已存在
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "⚠️  环境 '$ENV_NAME' 已存在"
    read -p "是否删除并重新创建? [y/N]: " recreate
    if [[ "$recreate" =~ ^[Yy]$ ]]; then
        echo "删除现有环境..."
        conda env remove -n "$ENV_NAME" -y
    else
        echo "使用现有环境"
        echo ""
        echo "激活环境:"
        echo "  conda activate $ENV_NAME"
        echo ""
        echo "如果需要安装CUDA版本的PyTorch，请运行:"
        echo "  pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu118"
        exit 0
    fi
fi

# 创建环境
echo "创建conda环境..."
cd "$SCRIPT_DIR"
conda env create -f "$ENV_FILE" -n "$ENV_NAME"

echo ""
echo "✅ 环境创建成功！"
echo ""

# 如果使用CUDA版本，提示安装PyTorch
if [ "$ENV_FILE" = "environment_cuda.yml" ]; then
    echo "⚠️  注意: 需要手动安装CUDA版本的PyTorch"
    echo ""
    echo "请运行以下命令激活环境并安装PyTorch:"
    echo ""
    echo "  conda activate $ENV_NAME"
    echo ""
    echo "然后根据CUDA版本选择安装命令:"
    echo ""
    echo "  # CUDA 11.8:"
    echo "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    echo ""
    echo "  # CUDA 12.1 (最低版本2.1.0):"
    echo "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    echo ""
    echo "  # 或指定具体版本（CUDA 12.1推荐2.1.2）:"
    echo "  pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121"
    echo ""
else
    echo "激活环境:"
    echo "  conda activate $ENV_NAME"
    echo ""
    echo "验证安装:"
    echo "  python -c \"import torch; print(f'PyTorch版本: {torch.__version__}')\""
    echo "  python -c \"import torch; print(f'CUDA可用: {torch.cuda.is_available()}')\""
fi

echo ""
echo "=========================================="
echo "安装完成！"
echo "=========================================="

