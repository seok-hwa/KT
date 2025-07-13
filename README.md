# Knowledge Tailoring (KT)

Official PyTorch implementation of:

**Knowledge Tailoring: Bridging the Teacher-Student Gap in Semantic Segmentation**  

---

## Overview

Knowledge Tailoring (KT) is a novel knowledge distillation framework designed to mitigate the **teacher-student gap** in semantic segmentation.  
It introduces two key modules:

- **Feature Tailoring (FT):** Tailors teacher features to match the capacity of the student, alleviating over-complex representations.
- **Logit Tailoring (LT):** Refines teacher logits to reduce uncertainty and provide clearer supervision for boundary regions.

KT effectively improves student performance without requiring extra models or significant computational overhead.

---

## Highlights

- Tailored knowledge distillation for both encoder and decoder in semantic segmentation.
- Addresses both capacity gap and task difficulty in a unified framework.
- Compatible with standard segmentation architectures (e.g., SegFormer, DeepLabV3+).
- Achieves state-of-the-art results on standard benchmarks.

---

## Requirement

Ubuntu 18.04 LTS

Python 3.8 (Anaconda is recommended)

CUDA 11.4

PyTorch 1.8.0

NCCL for CUDA 11.4

Install python packages:
```bash
pip install timm==0.3.2
pip install mmcv-full==1.2.7
pip install opencv-python==4.5.1.48
```

---

## Training
Example (Cityscapes)

```bash
sh train_KT.sh
```

---

## Results
| Method                     | Params (M) | FLOPs (G) | Val mIoU (%) |
|----------------------------|------------|-----------|--------------|
| **T: DeepLabV3-ResNet101** | 61.1M      | 687.8G    | 78.07        |
| **S: DeepLabV3-ResNet18**  | 13.6M      | 159.0G    | 74.21        |
| + SKD                      |            |           | 75.42        |
| + IFVD                     |            |           | 75.59        |
| + CWD                      |            |           | 75.55        |
| + CIRKD                    |            |           | 76.38        |
| + DIST                     |            |           | 77.10        |
| + MasKD                    |            |           | 77.00        |
| + DiffKD                   |            |           | 77.78        |
| + LAD                      |            |           | 75.60        |
| + LAD*                     |            |           | 77.70        |
|----------------------------|------------|-----------|--------------|
| + **KT (Ours)**            | 13.6M      | 159.0G    | 77.98        |
