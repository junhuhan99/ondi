# OnDi - On-Device AI Model

> **100% From Scratch | 100% Owned License | Coding & English Specialized**

OnDi는 처음부터 직접 설계하고 학습한 커스텀 AI 모델입니다. 코딩과 영어에 특화되어 있으며, 온디바이스 배포를 목표로 설계되었습니다.

## Features

- **100% Custom Architecture**: Transformer 모델을 처음부터 직접 설계
- **100% Owned License**: 모든 코드와 모델 가중치에 대한 완전한 소유권
- **Coding Specialized**: Python, JavaScript 등 프로그래밍 코드 생성
- **English Specialized**: 자연스러운 영어 텍스트 생성
- **On-Device Ready**: 경량화된 모델로 로컬 실행 가능

## Model Architecture

```
OnDi Model (GPT-style Decoder-only Transformer)
├── Token Embedding
├── Position Embedding
├── Transformer Blocks (x8-12)
│   ├── Multi-Head Self-Attention
│   ├── Layer Normalization (Pre-norm)
│   └── Feed-Forward Network (GELU)
├── Final Layer Normalization
└── Language Model Head (weight-tied)
```

### Model Sizes

| Size | Parameters | d_model | Layers | Heads | Context |
|------|------------|---------|--------|-------|---------|
| Tiny | ~15M | 256 | 4 | 4 | 512 |
| Small | ~85M | 512 | 8 | 8 | 1024 |
| Medium | ~150M | 768 | 12 | 12 | 1024 |

## Installation

```bash
# Clone repository
git clone https://github.com/junhuhan99/ondi.git
cd ondi

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Training

### Quick Start

```bash
# Train small model (recommended for most cases)
python train.py --model_size small --max_steps 50000

# Train with custom settings
python train.py \
    --model_size medium \
    --batch_size 8 \
    --max_steps 100000 \
    --learning_rate 3e-4 \
    --coding_samples 50000 \
    --english_samples 50000
```

### Resume Training

```bash
python train.py --resume ./checkpoints/checkpoint_step_10000
```

## Inference

### Command Line

```bash
# Single prompt
python inference.py --prompt "def hello_world():" --max_tokens 200

# Interactive mode
python inference.py --interactive
```

### Python API

```python
from inference import OnDiInference

# Load model
model = OnDiInference("./checkpoints/final")

# Generate text
response = model.generate(
    prompt="def fibonacci(n):",
    max_new_tokens=200,
    temperature=0.8
)
print(response)
```

## Project Structure

```
ondi/
├── src/
│   ├── model.py        # Transformer model architecture
│   ├── tokenizer.py    # BPE tokenizer implementation
│   └── dataset.py      # Dataset preparation
├── train.py            # Training script
├── inference.py        # Inference script
├── requirements.txt    # Dependencies
└── README.md           # Documentation
```

## Training Data

The model is trained on:
- **Coding Data**: Python, JavaScript code from open sources
- **English Data**: Wikipedia, web text

All training data is from publicly available sources with permissive licenses.

## Hardware Requirements

### Training
- GPU: NVIDIA T4 (16GB) or better
- RAM: 32GB+
- Storage: 100GB+

### Inference
- CPU: Any modern CPU
- RAM: 4GB+ (for small model)
- GPU: Optional (for faster inference)

## License

**This project is 100% owned by the creator.**

All code, model architecture, and trained weights are original work and fully owned by the repository owner. You may use, modify, and distribute this project according to your needs.

## Technical Details

### Tokenizer
- Type: Byte-Pair Encoding (BPE)
- Vocabulary Size: 32,000 tokens
- Special Tokens: `<pad>`, `<unk>`, `<bos>`, `<eos>`, `<code>`, `</code>`

### Training
- Optimizer: AdamW (β1=0.9, β2=0.95)
- Learning Rate: 3e-4 with warmup and cosine decay
- Weight Decay: 0.1
- Gradient Clipping: 1.0
- Mixed Precision: FP16

### Architecture Features
- Pre-LayerNorm (more stable training)
- GELU activation
- Weight tying (embedding ↔ output)
- Rotary-style position embeddings

## Citation

```bibtex
@software{ondi2024,
  title = {OnDi: On-Device AI Model for Coding and English},
  author = {Jun Hu Han},
  year = {2024},
  url = {https://github.com/junhuhan99/ondi}
}
```

## Acknowledgments

- Built with PyTorch
- Inspired by GPT and LLaMA architectures
- Trained on AWS EC2 with NVIDIA T4 GPU
