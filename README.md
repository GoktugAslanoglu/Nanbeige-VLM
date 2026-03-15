# Nanbeige4.1-VLM

SigLIP so400m vision encoder + 2-layer MLP projector + Nanbeige4.1-3B LLM.

Stage 1 pretrain checkpoint trained on LLaVA-CC3M-Pretrain-595K (595K image-caption pairs).

---

## Installation

```bash
pip install transformers torch accelerate sentencepiece safetensors Pillow
```

---

## Usage

```python
from transformers import AutoModel, AutoTokenizer
from PIL import Image

# Load
model     = AutoModel.from_pretrained("SkyAsl/Nanbeige4.1-VLM-Base", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("SkyAsl/Nanbeige4.1-VLM-Base", trust_remote_code=True)
model.set_tokenizer(tokenizer)

# Describe an image
image  = Image.open("photo.jpg")
result = model.describe(image)
print(result)

# Custom prompt
result = model.describe(image, prompt="What objects are in this image?")
print(result)

# Creative sampling
result = model.describe(image, do_sample=True, temperature=0.7)
print(result)
```

---

## Model Details

| Component | Model | Status |
|---|---|---|
| Vision encoder | google/siglip-so400m-patch14-384 | Frozen |
| Language model | Nanbeige/Nanbeige4.1-3B | Frozen |
| Projector | 2-layer MLP + 2×2 avg-pool (729→196 tokens) | Trained |

- **Dataset:** liuhaotian/LLaVA-CC3M-Pretrain-595K
- **Hardware:** A100 80GB
- **Training time:** ~6 hours
- **Final loss:** ~2.47

---

## ⚠️ Stage 1 Only

This is a Stage 1 pretrain checkpoint. The model can describe image content
but cannot answer questions or follow complex instructions yet.
Stage 2 instruction fine-tuning is in progress.
