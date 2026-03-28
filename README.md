# CA-LLE
# Enhanced Low-Light Image Enhancement with Cognitive Awareness

基于认知感知的增强型低光图像增强模型，结合CLIP语义编码器和自适应调制机制，实现高质量的低光图像增强。

## 特性
- 结合CLIP语义编码器的认知感知增强
- 自适应FiLM层和多尺度注意力门控
- 多损失函数融合（重建、颜色、平滑度）
- 自适应早停机制和学习率调度
- 支持有监督和自监督训练

## 环境配置
```bash
pip install -r requirements.txt
