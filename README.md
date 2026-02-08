# EuroSAT Land Use Classification

Deep learning-based land use and land cover (LULC) classification using transfer learning on the EuroSAT benchmark dataset.

## Overview

This project implements an image classification pipeline for satellite imagery using the EuroSAT dataset, consisting of 27,000 labeled Sentinel-2 satellite image patches. The model leverages transfer learning with a pre-trained ResNet50 backbone, fine-tuned for 10-class LULC classification, achieving **97.08% test accuracy**.

**Land Use Classes**: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake

## Performance Metrics

| Metric | Score |
|--------|-------|
| **Test Accuracy** | **97.08%** |
| **Validation Accuracy** | **97.10%** |
| **Training Accuracy** | **97.83%** |
| **Training Time** | ~12 min (10 epochs, GPU) |

### Training Progression

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|------------|-----------|----------|---------|
| 1 | 0.4753 | 84.55% | 0.1832 | 93.50% |
| 5 | 0.1526 | 94.87% | 0.1354 | 95.20% |
| 10 | 0.0623 | 97.83% | 0.0896 | **97.10%** |

The model demonstrates strong convergence with minimal overfitting (train-val gap < 1%), indicating effective regularization through transfer learning.

## Architecture

**Model**: ResNet50 (ImageNet pre-trained) + Custom FC Head  
**Training Strategy**: Feature extraction with frozen convolutional base  
**Framework**: PyTorch with torchvision

```
ResNet50 (frozen)
    ↓
Global Average Pooling
    ↓
Flatten
    ↓
Linear(2048 → 10)
    ↓
Softmax
```

## Technical Stack

```bash
torch >= 2.0.0
torchvision >= 0.15.0
numpy
pandas
matplotlib
tqdm
```

## Implementation Details

### Data Pipeline
- **Input Resolution**: 64×64 RGB
- **Normalization**: ImageNet statistics (μ=[0.485, 0.456, 0.406], σ=[0.229, 0.224, 0.225])
- **Train/Val/Test Split**: 64%/16%/20%
- **Batch Size**: 32

### Training Configuration
```python
Optimizer: Adam (lr=1e-4)
Scheduler: CosineAnnealingLR (η_min=1e-6)
Loss Function: CrossEntropyLoss
Epochs: 10
Device: CUDA (if available)
```

### Transfer Learning Approach
- Initialized with ResNet50 ImageNet1K-V1 weights
- Froze all convolutional layers (feature extractor mode)
- Trained only the final classification head
- Applied cosine annealing for learning rate decay

## Usage

### Setup
```bash
git clone https://github.com/Tanishq-Mehta-1/LULC-EUROSAT.git
cd LULC-EUROSAT
pip install -r requirements.txt
```

### Training
```python
# Run in Jupyter/Colab - dataset auto-downloads
jupyter notebook LULC_EUROSAT.ipynb
```

Model checkpointing: Best weights saved based on validation accuracy as `best_lulc_model_10epochs.pth`

## Dataset Citation

```bibtex
@article{helber2019eurosat,
  title={EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification},
  author={Helber, Patrick and Bischke, Benjamin and Dengel, Andreas and Borth, Damian},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  volume={12},
  number={7},
  pages={2217--2226},
  year={2019},
  publisher={IEEE}
}
```

## References

- [EuroSAT Dataset](https://github.com/phelber/EuroSAT)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
