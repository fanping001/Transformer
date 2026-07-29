🧠 Transformer 从零实现
手动实现 Transformer 模型，深入理解 Attention Is All You Need 的核心机制

![Python]https://img.shields.io/badge/Python-3.8+-blue.svg
https://img.shields.io/badge/PyTorch-2.0+-red.svg

📖 项目简介
本项目用纯 PyTorch 从零手动实现 Transformer 模型，不依赖预训练权重或高级封装库（如 Hugging Face Transformers）。代码逐行实现以下核心组件：

✅ Multi-Head Self-Attention（多头自注意力机制）

✅ Positional Encoding（位置编码）

✅ Feed-Forward Network（前馈神经网络）

✅ Encoder & Decoder 完整结构

✅ Masking（Padding Mask + Look-ahead Mask）

✅ 完整的训练与推理流程

目标：通过手动实现，彻底搞懂 Transformer 的每一个细节。

🏗️ 代码结构
text
Transformer/
├── models/                     # 模型核心实现
│   ├── __init__.py
│   ├── attention.py           # 多头注意力机制
│   ├── positional_encoding.py # 位置编码
│   ├── feedforward.py         # 前馈网络
│   ├── encoder.py             # Encoder 层和完整 Encoder
│   ├── decoder.py             # Decoder 层和完整 Decoder
│   └── transformer.py         # 完整 Transformer 模型
├── utils/                     # 工具函数
│   ├── __init__.py
│   ├── data_utils.py          # 数据加载与预处理
│   └── masks.py               # 各种 Mask 生成
├── train.py                   # 训练脚本
├── inference.py               # 推理脚本
├── config.py                  # 模型配置参数
├── requirements.txt           # 依赖列表
└── README.md                  # 项目说明
🚀 快速开始
环境准备
bash
# 克隆项目
git clone https://github.com/fanping001/Transformer.git
cd Transformer

# 安装依赖
pip install -r requirements.txt
依赖列表 (requirements.txt)
text
torch>=2.0.0
numpy>=1.24.0
tqdm>=4.65.0
matplotlib>=3.7.0  # 用于可视化
🎯 核心组件说明
1. Multi-Head Self-Attention
python
# 使用示例
from models.attention import MultiHeadAttention

attn = MultiHeadAttention(d_model=512, num_heads=8)
output = attn(query, key, value, mask=None)
2. Positional Encoding
采用正弦/余弦函数生成位置编码：

text
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

3. 完整模型
python
from models.transformer import Transformer

model = Transformer(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    d_model=512,
    num_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    d_ff=2048,
    max_len=512,
    dropout=0.1
)
🏃 训练与推理
训练模型
bash
python train.py \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --data_path ./data/
推理（翻译/生成）
bash
python inference.py \
    --model_path ./checkpoints/best_model.pth \
    --src_sentence "Hello, world!"
📊 可视化支持
项目提供了注意力权重可视化工具，帮助你理解 Attention 的工作机制：

python
from utils.visualization import plot_attention

# 可视化编码器自注意力权重
plot_attention(attention_weights, src_tokens)
📚 学习资源
📄 Attention Is All You Need - 原始论文

🎥 The Illustrated Transformer - 图解 Transformer

📖 Harvard NLP 实现 - 经典教学实现

🛠️ 后续计划
□ 添加单元测试
□ 支持更多数据集（IWSLT, WMT）
□ 添加 TensorBoard 训练监控
□ 支持模型并行训练
□ 导出 ONNX 模型
🤝 贡献
欢迎提交 Issue 和 Pull Request！

Fork 本项目

创建你的功能分支 (git checkout -b feature/AmazingFeature)

提交更改 (git commit -m 'Add some AmazingFeature')

推送到分支 (git push origin feature/AmazingFeature)

打开一个 Pull Request

📄 许可证
本项目仅供学习交流使用，采用 MIT 许可证。

✨ 致谢
感谢 Attention Is All You Need 论文作者的开创性工作

感谢 PyTorch 社区提供的优秀工具

⭐ 如果这个项目对你有帮助，欢迎点个 Star！
