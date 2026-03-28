#!/bin/bash
# 推理脚本

# 设置CUDA可见设备
export CUDA_VISIBLE_DEVICES=0

# 推理参数
CKPT_PATH="./runs/enhanced_exp/best_enhanced.ckpt"
INPUT_DIR="./data/test/low"
SAVE_DIR="./outputs/enhanced"
IMAGE_SIZE=256

# 创建保存目录
mkdir -p $SAVE_DIR

# 运行推理
python main_enhanced.py --mode infer \
    --ckpt $CKPT_PATH \
    --input_dir $INPUT_DIR \
    --save_dir $SAVE_DIR \
    --size $IMAGE_SIZE

echo "推理完成！结果保存在: $SAVE_DIR"
