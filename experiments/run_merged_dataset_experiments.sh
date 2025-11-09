#!/bin/bash

# 混合数据集上的三组对比实验
# 实验1: BP batch_size=2, ZO 64方向，每个方向独立batch_size=2，数据不重合（分割数据集）
# 实验2: BP batch_size=2, ZO 64方向，共享128条数据
# 实验3: 纯BP训练

BASE_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$BASE_DIR"

SCRIPT="core/reproduce_zo_paper_1106.py"
DATASET="dclm-pubmedqa-merged"  # 混合数据集

# 通用参数
MODEL_SIZE="200M"
EPOCHS=2
EVAL_INTERVAL=100
SCOPE="full"
LEARNING_RATE=0.001
BP_INTERVAL=1
OPTIMIZER="mudamw"

# GPU分配：3个实验使用GPU 0, 1, 2
GPUS=(0 1 2)

# 创建日志目录
LOG_DIR="experiment_logs/merged_dataset_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "混合数据集对比实验"
echo "=========================================="
echo "数据集: $DATASET"
echo "日志目录: $LOG_DIR"
echo ""

# 实验1: BP batch_size=2, ZO 64方向，每个方向独立batch_size=2，数据不重合
echo "实验1: BP batch_size=2, ZO 64方向独立数据（分割数据集）"
echo "  - BP: batch_size=2"
echo "  - ZO: 64方向 × batch_size=2 = 128条数据"
echo "  - 数据不重合（分割数据集）"
echo ""

EXP1_NAME="exp1_bp2_zo64_split"
EXP1_LOG="$LOG_DIR/${EXP1_NAME}.log"
EXP1_CSV="logs/${EXP1_NAME}_$(date +%Y%m%d_%H%M%S)/training_log.csv"

nohup bash -c "
    source /opt/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
    conda activate zo_withbp
    CUDA_VISIBLE_DEVICES=${GPUS[0]} python $SCRIPT \
        --mode Instruct \
        --scope $SCOPE \
        --query_budget_q 64 \
        --learning_rate $LEARNING_RATE \
        --batch_size 1 \
        --bp_batch_size 2 \
        --zo_query_batch_size 2 \
        --epochs $EPOCHS \
        --eval_interval $EVAL_INTERVAL \
        --optimizer $OPTIMIZER \
        --bp_interval $BP_INTERVAL \
        --dataset $DATASET \
        --split_dataset_for_bp_zo \
        --model_size $MODEL_SIZE \
        --csv_file $EXP1_CSV \
        --run_name $EXP1_NAME \
        > $EXP1_LOG 2>&1
" &

PID1=$!
echo "  实验1 PID: $PID1, GPU: ${GPUS[0]}, 日志: $EXP1_LOG"
echo ""

# 实验2: BP batch_size=2, ZO 64方向，共享128条数据（但BP和ZO数据不重合）
echo "实验2: BP batch_size=2, ZO 64方向共享数据（BP和ZO数据分割）"
echo "  - BP: batch_size=2（使用数据集的一部分）"
echo "  - ZO: 64方向共享128条数据（使用数据集的另一部分，不重合）"
echo "  注意: zo_query_batch_size=2, 64方向 × 2 = 128条，所有ZO方向共享，但与BP数据不重合"
echo ""

EXP2_NAME="exp2_bp2_zo64_shared"
EXP2_LOG="$LOG_DIR/${EXP2_NAME}.log"
EXP2_CSV="logs/${EXP2_NAME}_$(date +%Y%m%d_%H%M%S)/training_log.csv"

nohup bash -c "
    source /opt/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
    conda activate zo_withbp
    CUDA_VISIBLE_DEVICES=${GPUS[1]} python $SCRIPT \
        --mode Instruct \
        --scope $SCOPE \
        --query_budget_q 64 \
        --learning_rate $LEARNING_RATE \
        --batch_size 1 \
        --bp_batch_size 2 \
        --zo_query_batch_size 2 \
        --epochs $EPOCHS \
        --eval_interval $EVAL_INTERVAL \
        --optimizer $OPTIMIZER \
        --bp_interval $BP_INTERVAL \
        --dataset $DATASET \
        --split_dataset_for_bp_zo \
        --zo_share_data_across_queries \
        --model_size $MODEL_SIZE \
        --csv_file $EXP2_CSV \
        --run_name $EXP2_NAME \
        > $EXP2_LOG 2>&1
" &

PID2=$!
echo "  实验2 PID: $PID2, GPU: ${GPUS[1]}, 日志: $EXP2_LOG"
echo ""

# 实验3: 纯BP训练
echo "实验3: 纯BP训练（基线）"
echo "  - BP: batch_size=2"
echo "  - 不使用ZO"
echo ""

EXP3_NAME="exp3_bp_only"
EXP3_LOG="$LOG_DIR/${EXP3_NAME}.log"
EXP3_CSV="logs/${EXP3_NAME}_$(date +%Y%m%d_%H%M%S)/training_log.csv"

nohup bash -c "
    source /opt/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
    conda activate zo_withbp
    CUDA_VISIBLE_DEVICES=${GPUS[2]} python $SCRIPT \
        --mode FO \
        --scope $SCOPE \
        --learning_rate $LEARNING_RATE \
        --batch_size 2 \
        --epochs $EPOCHS \
        --eval_interval $EVAL_INTERVAL \
        --optimizer $OPTIMIZER \
        --dataset $DATASET \
        --model_size $MODEL_SIZE \
        --csv_file $EXP3_CSV \
        --run_name $EXP3_NAME \
        > $EXP3_LOG 2>&1
" &

PID3=$!
echo "  实验3 PID: $PID3, GPU: ${GPUS[2]}, 日志: $EXP3_LOG"
echo ""

echo "=========================================="
echo "所有实验已启动"
echo "=========================================="
echo "实验1 (分割数据): PID $PID1, GPU ${GPUS[0]}"
echo "实验2 (共享数据): PID $PID2, GPU ${GPUS[1]}"
echo "实验3 (纯BP):     PID $PID3, GPU ${GPUS[2]}"
echo ""
echo "查看日志:"
echo "  tail -f $EXP1_LOG"
echo "  tail -f $EXP2_LOG"
echo "  tail -f $EXP3_LOG"
echo ""
echo "查看进程:"
echo "  ps aux | grep $SCRIPT"
echo ""
echo "等待所有实验完成..."
echo ""

# 等待所有后台进程完成
wait $PID1
STATUS1=$?
wait $PID2
STATUS2=$?
wait $PID3
STATUS3=$?

echo ""
echo "=========================================="
echo "实验完成状态"
echo "=========================================="
if [ $STATUS1 -eq 0 ]; then
    echo "✅ 实验1 (分割数据): 成功"
else
    echo "❌ 实验1 (分割数据): 失败 (退出码: $STATUS1)"
fi

if [ $STATUS2 -eq 0 ]; then
    echo "✅ 实验2 (共享数据): 成功"
else
    echo "❌ 实验2 (共享数据): 失败 (退出码: $STATUS2)"
fi

if [ $STATUS3 -eq 0 ]; then
    echo "✅ 实验3 (纯BP): 成功"
else
    echo "❌ 实验3 (纯BP): 失败 (退出码: $STATUS3)"
fi

echo ""
echo "CSV日志文件:"
echo "  $EXP1_CSV"
echo "  $EXP2_CSV"
echo "  $EXP3_CSV"
echo ""
echo "可以使用以下命令绘制对比图:"
echo "  python visualization/plot_two_experiments.py"
echo ""

