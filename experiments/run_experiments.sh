#!/bin/bash

# 多数据集训练实验脚本
# 所有实验使用: batch_size=2, q=64, optimizer=mudamw, model=200M, epochs=2, eval_interval=100
# 6个实验将在GPU 2-7上并行执行

BASE_DIR=$(cd "$(dirname "$0")/.." && pwd)
SCRIPT="core/reproduce_zo_paper_1105.py"

# 通用参数
BATCH_SIZE=2
Q=64
OPTIMIZER="mudamw"
MODEL_SIZE="200M"
EPOCHS=2
EVAL_INTERVAL=100
SCOPE="full"

# GPU分配：6个实验使用GPU 2-7
GPUS=(2 3 4 5 6 7)

# 创建日志目录
LOG_DIR="experiment_logs"
mkdir -p $LOG_DIR

# 激活conda环境
source /data/cdq/.conda/etc/profile.d/conda.sh
conda activate speechbrain

echo "=========================================="
echo "开始运行多数据集训练实验（并行执行）"
echo "使用GPU: ${GPUS[@]}"
echo "日志目录: $LOG_DIR"
echo "=========================================="

# 存储所有后台任务的PID
PIDS=()

# ============================================
# 组1: dclm + cosmopedia
# ============================================

echo ""
echo "=========================================="
echo "组1: dclm + cosmopedia 实验"
echo "=========================================="

# 实验1: FO模式 - 在dclm上训练，在cosmopedia评估loss
echo ""
echo "实验1: FO模式 - dclm训练 + cosmopedia评估 (GPU ${GPUS[0]})"
CUDA_VISIBLE_DEVICES=${GPUS[0]} python $SCRIPT \
    --mode FO \
    --scope $SCOPE \
    --learning_rate 0.001 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --optimizer $OPTIMIZER \
    --model_size $MODEL_SIZE \
    --dataset dclm-local \
    --eval_dataset cosmopedia-100k \
    --eval_interval $EVAL_INTERVAL \
    --log_interval 10 \
    --run_name "FO_dclm_train_cosmopedia_eval" \
    > $LOG_DIR/exp1_fo_dclm_cosmopedia.log 2>&1 &
PIDS+=($!)
echo "  启动在后台，PID: $!, GPU: ${GPUS[0]}, 日志: $LOG_DIR/exp1_fo_dclm_cosmopedia.log"

# 实验2: Instruct模式 - FO在dclm上生成梯度，指导ZO在cosmopedia上的梯度方向
echo ""
echo "实验2: Instruct模式 - BP用dclm，训练用cosmopedia (GPU ${GPUS[1]})"
CUDA_VISIBLE_DEVICES=${GPUS[1]} python $SCRIPT \
    --mode Instruct \
    --scope $SCOPE \
    --query_budget_q $Q \
    --learning_rate 0.001 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --optimizer $OPTIMIZER \
    --model_size $MODEL_SIZE \
    --dataset cosmopedia-100k \
    --bp_dataset dclm-local \
    --bp_interval 1 \
    --eval_interval $EVAL_INTERVAL \
    --log_interval 10 \
    --run_name "Instruct_cosmopedia_train_dclm_bp" \
    > $LOG_DIR/exp2_instruct_cosmopedia_dclm.log 2>&1 &
PIDS+=($!)
echo "  启动在后台，PID: $!, GPU: ${GPUS[1]}, 日志: $LOG_DIR/exp2_instruct_cosmopedia_dclm.log"

# 实验3: ZO模式 - ZO在cosmopedia训练
echo ""
echo "实验3: ZO模式 - cosmopedia训练 (GPU ${GPUS[2]})"
CUDA_VISIBLE_DEVICES=${GPUS[2]} python $SCRIPT \
    --mode ZO \
    --scope $SCOPE \
    --query_budget_q $Q \
    --learning_rate 0.001 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --optimizer $OPTIMIZER \
    --model_size $MODEL_SIZE \
    --dataset cosmopedia-100k \
    --eval_interval $EVAL_INTERVAL \
    --log_interval 10 \
    --run_name "ZO_cosmopedia_train" \
    > $LOG_DIR/exp3_zo_cosmopedia.log 2>&1 &
PIDS+=($!)
echo "  启动在后台，PID: $!, GPU: ${GPUS[2]}, 日志: $LOG_DIR/exp3_zo_cosmopedia.log"

# ============================================
# 组2: dclm + pubmedqa
# ============================================

echo ""
echo "=========================================="
echo "组2: dclm + pubmedqa 实验"
echo "=========================================="

# 实验4: FO模式 - 在dclm上训练，在pubmedqa评估loss
echo ""
echo "实验4: FO模式 - dclm训练 + pubmedqa评估 (GPU ${GPUS[3]})"
CUDA_VISIBLE_DEVICES=${GPUS[3]} python $SCRIPT \
    --mode FO \
    --scope $SCOPE \
    --learning_rate 0.001 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --optimizer $OPTIMIZER \
    --model_size $MODEL_SIZE \
    --dataset dclm-local \
    --eval_dataset pubmedqa-local \
    --eval_interval $EVAL_INTERVAL \
    --log_interval 10 \
    --run_name "FO_dclm_train_pubmedqa_eval" \
    > $LOG_DIR/exp4_fo_dclm_pubmedqa.log 2>&1 &
PIDS+=($!)
echo "  启动在后台，PID: $!, GPU: ${GPUS[3]}, 日志: $LOG_DIR/exp4_fo_dclm_pubmedqa.log"

# 实验5: Instruct模式 - FO在dclm上生成梯度，指导ZO在pubmedqa上的梯度方向
echo ""
echo "实验5: Instruct模式 - BP用dclm，训练用pubmedqa (GPU ${GPUS[4]})"
CUDA_VISIBLE_DEVICES=${GPUS[4]} python $SCRIPT \
    --mode Instruct \
    --scope $SCOPE \
    --query_budget_q $Q \
    --learning_rate 0.001 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --optimizer $OPTIMIZER \
    --model_size $MODEL_SIZE \
    --dataset pubmedqa-local \
    --bp_dataset dclm-local \
    --bp_interval 1 \
    --eval_interval $EVAL_INTERVAL \
    --log_interval 10 \
    --run_name "Instruct_pubmedqa_train_dclm_bp" \
    > $LOG_DIR/exp5_instruct_pubmedqa_dclm.log 2>&1 &
PIDS+=($!)
echo "  启动在后台，PID: $!, GPU: ${GPUS[4]}, 日志: $LOG_DIR/exp5_instruct_pubmedqa_dclm.log"

# 实验6: ZO模式 - ZO在pubmedqa训练
echo ""
echo "实验6: ZO模式 - pubmedqa训练 (GPU ${GPUS[5]})"
CUDA_VISIBLE_DEVICES=${GPUS[5]} python $SCRIPT \
    --mode ZO \
    --scope $SCOPE \
    --query_budget_q $Q \
    --learning_rate 0.001 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --optimizer $OPTIMIZER \
    --model_size $MODEL_SIZE \
    --dataset pubmedqa-local \
    --eval_interval $EVAL_INTERVAL \
    --log_interval 10 \
    --run_name "ZO_pubmedqa_train" \
    > $LOG_DIR/exp6_zo_pubmedqa.log 2>&1 &
PIDS+=($!)
echo "  启动在后台，PID: $!, GPU: ${GPUS[5]}, 日志: $LOG_DIR/exp6_zo_pubmedqa.log"

echo ""
echo "=========================================="
echo "所有6个实验已在后台启动！"
echo "=========================================="
echo ""
echo "实验状态："
for i in "${!PIDS[@]}"; do
    EXP_NUM=$((i+1))
    GPU_IDX=${GPUS[$i]}
    PID=${PIDS[$i]}
    echo "  实验${EXP_NUM} (GPU ${GPU_IDX}): PID ${PID}"
done

echo ""
echo "等待所有实验完成..."
echo "可以使用以下命令监控："
echo "  tail -f $LOG_DIR/exp*_*.log"
echo "  或"
echo "  watch -n 5 'tail -20 $LOG_DIR/exp*_*.log'"
echo ""

# 等待所有后台任务完成
for PID in "${PIDS[@]}"; do
    wait $PID
    STATUS=$?
    if [ $STATUS -eq 0 ]; then
        echo "✅ PID $PID 完成"
    else
        echo "❌ PID $PID 失败，退出码: $STATUS"
    fi
done

echo ""
echo "=========================================="
echo "所有实验完成！"
echo "=========================================="
echo ""
echo "查看日志："
echo "  ls -lh $LOG_DIR/"
echo ""

