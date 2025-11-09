#!/bin/bash

# 后台运行 Instruct 模式，25000 步
# 参数：bs=2, q=64, bp_interval=1, mudamw, lr=1e-3
# 注意：由于没有 max_steps 参数，使用 epochs 控制
# 数据集约 55634 个样本，batch_size=2 => 每个epoch约27817步
# 设置 epochs=1 可达到约27817步（超过25000步）

GPU_ID=5
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs"
mkdir -p ${LOG_DIR}

LOG_FILE="${LOG_DIR}/instruct_bs2_q64_bp1_mudamw_lr1e3_25000steps_${TIMESTAMP}.log"
CSV_FILE="${LOG_DIR}/instruct_bs2_q64_bp1_mudamw_lr1e3_25000steps_${TIMESTAMP}.csv"

echo "Starting Instruct mode training..."
echo "GPU: ${GPU_ID}"
echo "Log file: ${LOG_FILE}"
echo "CSV file: ${CSV_FILE}"
echo "Command will run in background..."
echo ""
echo "Parameters:"
echo "  Mode: Instruct"
echo "  Batch size: 2"
echo "  Query budget (q): 64"
echo "  BP interval: 1"
echo "  Optimizer: mudamw"
echo "  Learning rate: 1e-3"
echo "  Dataset samples: ~55634"
echo "  Epochs: 1 (approx 27817 steps, will exceed 25000)"
echo ""

# 使用 nohup 在后台运行，输出到日志文件
nohup bash -c "source /data/lzy/conda/etc/profile.d/conda.sh && conda activate MeZO && CUDA_VISIBLE_DEVICES=${GPU_ID} python core/reproduce_zo_paper_new.py \
    --mode Instruct \
    --scope full \
    --batch_size 2 \
    --query_budget_q 64 \
    --bp_interval 1 \
    --optimizer mudamw \
    --learning_rate 1e-3 \
    --epochs 1 \
    --log_interval 1 \
    --disable_checkpoint \
    --csv_file ${CSV_FILE} \
    --run_name instruct_25000steps \
    > ${LOG_FILE} 2>&1" &

PID=$!
echo "Process started with PID: ${PID}"
echo ""
echo "To check progress: tail -f ${LOG_FILE}"
echo "To check if running: ps aux | grep ${PID}"
echo "To stop: kill ${PID}"

