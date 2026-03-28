#!/bin/bash
# 训练脚本

# 设置CUDA可见设备
export CUDA_VISIBLE_DEVICES=0

# 训练参数
TRAIN_LOW_DIR="./data/train/low"
TRAIN_GT_DIR="./data/train/gt"
VAL_LOW_DIR="./data/val/low"
VAL_GT_DIR="./data/val/gt"
OUT_DIR="./runs/enhanced_exp"
EPOCHS=100
BATCH_SIZE=8
LR=2e-4
VAL_INTERVAL=5

# 创建输出目录
mkdir -p $OUT_DIR

# 运行训练
python main_enhanced.py --mode train \
    --train_low $TRAIN_LOW_DIR \
    --train_gt $TRAIN_GT_DIR \
    --val_low $VAL_LOW_DIR \
    --val_gt $VAL_GT_DIR \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --val_interval $VAL_INTERVAL \
    --use_lpips 1 \
    --self_supervised 0 \
    --out_dir $OUT_DIR \
    --num_workers 4 \
    --seed 42

echo "训练完成！结果保存在: $OUT_DIR"
