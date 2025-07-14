# **Knowledge Tailoring: Bridging the Teacher-Student Gap in Semantic Segmentation**

> Seokhwa Cheung, Seungbeom Woo, Taehoon Kim, Wonjun Hwang

<div align=center><img src="https://github.com/seok-hwa/KT/blob/2e2a4f962abf9566b5cc68ec96df66059544279e/Figure.png" width="60%"></div><br/>

> **Abstract:** Knowledge distillation transfers knowledge from a high-capacity teacher network to a compact student network, but a large capacity gap often limits the student’s ability to fully benefit from the teacher’s guidance. In semantic segmentation, another major challenge is the difficulty in predicting accurate object boundaries, as even strong teacher models can produce ambiguous or imprecise outputs. To address both challenges, we present Knowledge Tailoring, a novel distillation framework that adapts the teacher’s knowledge to better match the student’s representational capacity and learning dynamics. Much like a tailor adjusts an oversized suit to fit the wearer’s shape, our method reshapes the teacher’s abundant but misaligned knowledge into a form more suitable for the student. KT introduces feature tailoring, which restruc- tures intermediate features based on channel-wise correlation to narrow the representation gap, and logit tailoring, which improves boundary prediction by refining class-specific logits. The tailoring strategy evolves throughout training, offering guidance that aligns with the student’s progress. Experiments on Cityscapes, Pascal VOC, and
ADE20K confirm that KT consistently enhances performance across a variety of architectures including DeepLabV3, PSPNet, and SegFormer

---

## Overview

Knowledge Tailoring (KT) is a novel knowledge distillation framework designed to mitigate the teacher-student gap in semantic segmentation.  
It introduces two key modules:

- Feature Tailoring (FT): Tailors teacher features to match the capacity of the student, alleviating over-complex representations.
- Logit Tailoring (LT): Refines teacher logits to reduce uncertainty and provide clearer supervision for boundary regions.


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

## Result
All models are trained over 8 * NVIDIA RTX A6000

### Performance on Cityscapes

| Method                     | Params (M) | FLOPs (G) | Val mIoU (%) |
|----------------------------|------------|-----------|--------------|
| **T: DeepLabV3-ResNet101** | 61.1M      | 687.8G    | [78.07](https://drive.google.com/file/d/1zUdhYPYCDCclWU3Wo7GbbTlM8ibQ_UC1/view?usp=sharing)        |
| **S: DeepLabV3-ResNet18**  | 13.6M      | 159.0G    | 74.21        |
| + **KT (Ours)**            | 13.6M      | 159.0G    | [77.98](https://drive.google.com/file/d/1QnNDCNMnqaUoc_n9p4opuoAYyeOfJaRw/view?usp=sharing)        |
| **S: PSPNet-ResNet18**     | 12.9M      | 131.7G    | 72.55        |
| + **KT (Ours)**            | 12.9M      | 131.7G    | [76.49](https://drive.google.com/file/d/1S1-R5pMF_zT4JR2vsYs1T9TmA1Ofk1xS/view?usp=sharing)        |
| **S: DeepLabV3-ResNet18**  | 3.2M       | 39.2G     | 73.12        |
| + **KT (Ours)**            | 3.2M       | 39.2G     | [76.17](https://drive.google.com/file/d/1TXjyKwI5WB97uSV4aHDY5knySVBXAimV/view?usp=sharing)        |

### Performance on Pascal VOC

| Method                     | Params (M) | FLOPs (G) | Val mIoU (%) |
|----------------------------|------------|-----------|--------------|
| **T: DeepLabV3-ResNet101** | 61.1M      | 384.84    | [77.67](https://drive.google.com/file/d/1rYTaVq_ooiAI4oFOcDP8K3SpSbjURGnX/view?usp=sharing)        |
| **S: DeepLabV3-ResNet18**  | 13.6M      | 86.0G     | 73.21        |
| + **KT (Ours)**            | 13.6M      | 86.0G     | [76.49](https://drive.google.com/file/d/1QnNDCNMnqaUoc_n9p4opuoAYyeOfJaRw/view?usp=sharing)        |
| **S: PSPNet-ResNet18**     | 12.9M      | 67.6G     | 73.33        |
| + **KT (Ours)**            | 12.9M      | 67.6G     | [75.52](https://drive.google.com/file/d/1S1-R5pMF_zT4JR2vsYs1T9TmA1Ofk1xS/view?usp=sharing)        | 

### Performance on ADE20K

| Method                     | Params (M) | FLOPs (G) | Val mIoU (%) |
|----------------------------|------------|-----------|--------------|
| **T: DeepLabV3-ResNet101** | 61.1M      | 384.4G    | 42.70        |
| **S: DeepLabV3-ResNet18**  | 13.6M      | 86.0G     | 32.91        |
| + **KT (Ours)**            | 13.6M      | 86.0G     | [35.64](https://drive.google.com/file/d/1QnNDCNMnqaUoc_n9p4opuoAYyeOfJaRw/view?usp=sharing)        |
| **S: PSPNet-ResNet18**     | 12.9M      | 67.6G     | 34.17        |
| + **KT (Ours)**            | 12.9M      | 67.6G     | [35.77](https://drive.google.com/file/d/1S1-R5pMF_zT4JR2vsYs1T9TmA1Ofk1xS/view?usp=sharing)        | 
