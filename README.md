# CA-LLE (`ca_lle` package)

**Cognitive-Aware Low-Light Enhancement** — modular PyTorch code: frozen **OpenCLIP** semantics + **U-Net** enhancement with FiLM and multi-scale attention.

## What lives here

| Module | Role |
|--------|------|
| `semantic_encoder.py` | Frozen CLIP ViT-B-32 image encoder |
| `conditioning.py` | FiLM modulation, semantic attention gates |
| `residual.py` | Channel attention, residual blocks |
| `unet.py` | Encoder–bottleneck–decoder with skip connections |
| `enhancer.py` | `EnhancedCognitiveAwareEnhancer`, `build_enhanced_model()` |
| `losses.py` | Supervised reconstruction/color/smoothness; self-supervised term |
| `dataset.py` | `EnhancedLowLightDataset` (paired or low-light-only) |
| `callbacks.py` | SSIM-based early stopping with warmup |
| `train_utils.py` | One epoch, loss aggregation, PSNR/SSIM/LPIPS validation |
| `trainer.py` | Full training loop (optimizer, scheduler, checkpoints) |
| `inference.py` | Run a checkpoint on a folder of images |

Public imports are re-exported from `ca_lle/__init__.py` (`__version__`, model builder, losses).

## Dependencies

See **`requirements.txt`** in this directory (same pins as the repository root). Install CUDA-enabled PyTorch from [pytorch.org](https://pytorch.org) if you use a GPU.

From the **parent** directory of this package (repository root):

```bash
pip install -r requirements.txt
```

Or, from the root, referencing this file:

```bash
pip install -r ca_lle/requirements.txt
```

## Usage

**Programmatic:**

```python
from ca_lle import build_enhanced_model, enhanced_loss_recon

model = build_enhanced_model(base_channels=32, use_attention=True)
```

**Command-line** (repo root): run `main_enhanced.py` — see the top-level **`README.md`** for training (`--mode train`) and inference (`--mode infer`), including low-light-only (self-supervised) training.

**Legacy import path:** `import model_enhanced` at the repo root still re-exports the same symbols.

## Citation / license

See the repository **README.md** for citation block and license notice. Third-party weights (OpenCLIP, etc.) follow their original licenses.
