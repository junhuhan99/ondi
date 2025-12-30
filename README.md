# OnDi - On-Device AI Model

> **100% From Scratch | 100% Owned License | Coding & English Specialized**

OnDi는 처음부터 직접 설계하고 학습한 커스텀 AI 모델입니다. 코딩과 영어에 특화되어 있으며, 온디바이스 배포를 목표로 설계되었습니다.

## Quick Start (바로 사용하기)

**학습 없이 바로 사용 가능!** 사전 학습된 모델이 포함되어 있습니다.

```bash
# 1. Clone
git clone https://github.com/junhuhan99/ondi.git
cd ondi

# 2. Install
pip install torch transformers

# 3. Run
python inference.py --checkpoint ./checkpoints/final --interactive
```

## Available Models

| Version | Parameters | Focus | Val Loss | Status |
|---------|------------|-------|----------|--------|
| **v1** | 26M | Coding + English | 0.0750 | ✅ Available |
| **v2** | 475M | Python 85% + English Conversation | - | 🔄 Training |

## Features

- **100% Custom Architecture**: Transformer 모델을 처음부터 직접 설계
- **100% Owned License**: 모든 코드와 모델 가중치에 대한 완전한 소유권
- **Pre-trained Weights Included**: 학습 없이 바로 사용 가능
- **Coding Specialized**: Python, JavaScript 등 프로그래밍 코드 생성
- **English Specialized**: 자연스러운 영어 텍스트 생성
- **On-Device Ready**: 경량화된 모델로 로컬 실행 가능

## Model Architecture

```
OnDi Model (GPT-style Decoder-only Transformer)
├── Token Embedding
├── Position Embedding
├── Transformer Blocks (x8-24)
│   ├── Multi-Head Self-Attention
│   ├── Layer Normalization (Pre-norm)
│   └── Feed-Forward Network (GELU)
├── Final Layer Normalization
└── Language Model Head (weight-tied)
```

### Model Configurations

| Size | Parameters | d_model | Layers | Heads | Context |
|------|------------|---------|--------|-------|---------|
| v1 (Small) | 26M | 512 | 8 | 8 | 1024 |
| v2 (Large) | 475M | 1280 | 24 | 20 | 1024 |

## Installation

```bash
# Clone repository
git clone https://github.com/junhuhan99/ondi.git
cd ondi

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Inference (추론)

```bash
# Interactive mode
python inference.py --checkpoint ./checkpoints/final --interactive

# Single prompt
python inference.py --checkpoint ./checkpoints/final --prompt "def hello_world():"
```

### Python API

```python
from inference import OnDiInference

# Load pre-trained model
model = OnDiInference("./checkpoints/final")

# Generate code
code = model.generate(
    prompt="def fibonacci(n):",
    max_new_tokens=200,
    temperature=0.8
)
print(code)

# Generate English text
text = model.generate(
    prompt="Machine learning is",
    max_new_tokens=100
)
print(text)
```

## Training (Optional)

직접 학습하고 싶다면:

### v1 Model (26M)

```bash
python train.py --model_size small --max_steps 30000
```

### v2 Model (475M) - Python 85% + English Conversation

```bash
python train_v2.py --batch_size 2 --max_steps 50000
```

## Project Structure

```
ondi/
├── src/
│   ├── model.py          # Transformer model architecture
│   ├── tokenizer.py      # BPE tokenizer implementation
│   ├── dataset.py        # Dataset preparation (v1)
│   └── dataset_v2.py     # Dataset preparation (v2: Python 85%)
├── checkpoints/
│   └── final/
│       ├── model.pt      # Pre-trained weights (26M)
│       ├── config.json   # Model configuration
│       └── tokenizer/    # Trained BPE tokenizer
├── train.py              # Training script (v1)
├── train_v2.py           # Training script (v2: 475M)
├── inference.py          # Inference script
├── requirements.txt      # Dependencies
└── README.md             # Documentation
```

## Training Details

### v1 Model
- **Data**: Coding + English mixed
- **Steps**: 30,000
- **Final Val Loss**: 0.0750
- **Training Time**: ~1 hour on T4

### v2 Model (In Progress)
- **Data**: Python 85% + English Conversation 15%
- **Steps**: 50,000
- **Parameters**: 475M
- **Expected Training Time**: ~6 hours on T4

## Hardware Requirements

### Inference
- CPU: Any modern CPU
- RAM: 2GB+ (v1), 4GB+ (v2)
- GPU: Optional (faster with CUDA)

### Training
- GPU: NVIDIA T4 (16GB) or better
- RAM: 32GB+
- Storage: 100GB+

## License

**This project is 100% owned by the creator.**

All code, model architecture, and trained weights are original work and fully owned by the repository owner (Jun Hu Han). You may use, modify, and distribute this project according to your needs.

## Technical Specifications

### Tokenizer
- Type: Byte-Pair Encoding (BPE)
- Vocabulary Size: ~1,000-32,000 tokens (varies by version)
- Special Tokens: `<pad>`, `<unk>`, `<bos>`, `<eos>`

### Training Configuration
- Optimizer: AdamW (β1=0.9, β2=0.95)
- Learning Rate: 2e-4 ~ 3e-4 with warmup and cosine decay
- Weight Decay: 0.1
- Gradient Clipping: 1.0
- Mixed Precision: FP16 (AMP)

### Architecture
- Pre-LayerNorm (stable training)
- GELU activation
- Weight tying (embedding ↔ output)
- Causal attention mask

## Citation

```bibtex
@software{ondi2024,
  title = {OnDi: On-Device AI Model for Coding and English},
  author = {Jun Hu Han},
  year = {2024},
  url = {https://github.com/junhuhan99/ondi}
}
```

## Author

**Jun Hu Han** (junhuhan99)

- Built with PyTorch
- Trained on AWS EC2 with NVIDIA T4 GPU
- 100% From Scratch Implementation
