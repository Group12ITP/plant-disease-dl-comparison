# Comparative Analysis of Deep Convolutional Architectures for Automated Plant Pathology Diagnosis and In-the-Wild Domain Generalization

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.15+](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://tensorflow.org/)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

**Module:** SE4050 – Deep Learning  
**Degree Program:** BSc (Hons) in Information Technology  
**Institution:** Sri Lanka Institute of Information Technology (SLIIT)  
**Academic Year:** 2026  
**Project Category:** Supervised Deep Learning (Multi-Class Computer Vision Classification)  

---

## 1. Project Overview

Automated visual disease classification enables early detection of crop pathogens, addressing agricultural losses that reduce global food yields by 20% to 40% annually. While deep learning models achieve near-perfect classification accuracies (>98%) on controlled laboratory datasets, practical field deployments consistently fail due to **shortcut learning** and domain shift.

This research project conducts an empirical comparison of four distinct deep learning paradigms:
1. **Custom 4-Block CNN** (trained from scratch; 285k parameters)
2. **ResNet50** (deep residual learning with identity shortcuts; 24.12M parameters)
3. **MobileNetV2** (inverted residual blocks with depthwise separable convolutions; 2.60M parameters)
4. **EfficientNetB0** (compound scaling with squeeze-and-excitation attention; 4.39M parameters)

All models were trained on the **PlantVillage** dataset ($N = 54,305$ images across 38 crop-disease classes) under a strict zero-data-leakage 80/10/10 stratified split, and subsequently stress-tested for out-of-distribution (OOD) generalization against the in-the-wild **PlantDoc** dataset ($N = 236$ images across 27 mapped classes). An ablation study demonstrates domain adaptation recovery via few-shot fine-tuning.

---

## 2. Repository Structure

```
├── Custom CNN.ipynb                         # Scratch 4-Block CNN pipeline & training
├── EDA.ipynb                                # Exploratory Data Analysis & visual profiling
├── EfficientNetB0.ipynb                     # EfficientNetB0 transfer learning pipeline
├── Mobilenetv2.ipynb                        # MobileNetV2 transfer learning pipeline
├── ResNet.ipynb                             # ResNet50 transfer learning pipeline
├── Model Comparison.ipynb                   # Cross-dataset evaluation on PlantDoc benchmark
├── Fine-Tuning EfficientNetB0 Ablation.ipynb# Few-shot domain adaptation ablation study
├── ROC-AUC.ipynb                            # Multiclass One-vs-Rest ROC-AUC & curves
├── Report_Draft.md                          # Complete 10-section academic project report
├── requirements.txt                         # Pinned dependency environment
├── README.md                                # Setup, replication, and project guide
└── results/
    ├── figures/                             # High-resolution evaluation charts & curves
    │   ├── eda_class_distribution.png       # 38-class frequency bar chart
    │   ├── eda_sample_images.png            # 4x4 visual morphological grid
    │   ├── roc_curves_plantvillage.png      # Comparative ROC curves (all 4 models)
    │   ├── plantdoc_per_class_accuracy.png  # Per-class OOD accuracy comparison
    │   ├── *_curves.png                     # Training and validation loss/acc curves
    │   └── *_confusion_matrix.png           # Raw and normalized confusion matrices
    ├── metrics/                             # Serialized JSON evaluation logs
    │   ├── *_history.json                   # Epoch-by-epoch loss, acc, lr, and time
    │   ├── *_metrics.json                   # Precision, recall, F1, latency, per-class
    │   ├── roc_auc_all_models.json          # Macro and weighted ROC-AUC scores
    │   ├── plantdoc_evaluation.json         # Zero-shot OOD evaluation scores
    │   └── plantdoc_finetuned_efficientnet.json # Few-shot ablation metrics
    ├── models/                              # Saved serialized Keras models (*.keras)
    └── splits/                              # Immutable CSV data manifests
        ├── class_names.csv                  # 38-class index mapping
        ├── train.csv                        # 43,444 stratified training samples
        ├── val.csv                          # 5,430 stratified validation samples
        └── test.csv                         # 5,431 stratified test samples
```

---

## 3. Dataset Access & Directory Setup

### 3.1 Primary Dataset: PlantVillage
- **Creator / Source:** David P. Hughes and Marcel Salathé (Penn State University & EPFL, 2015).
- **License:** CC BY-NC-SA 4.0.
- **Download Link:** [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease) or [spMohanty GitHub Repository](https://github.com/spMohanty/PlantVillage-Dataset).
- **Scale:** 54,305 RGB images across 38 crop-disease categories.

### 3.2 Secondary Benchmark: PlantDoc
- **Creator / Source:** Pratiksha Singh et al. (IIT Hyderabad & TCS Research, CoDS-COMAD 2020).
- **Download Link:** [PlantDoc Dataset on GitHub](https://github.com/pratikkayal/PlantDoc-Dataset).
- **Scale:** Real-world in-the-wild field imagery across 13 species and 30 disease classes.

### 3.3 Expected Local / Colab Directory Structure
Ensure data is organized as follows prior to execution:
```
data/
├── PlantVillage/
│   ├── Apple___Apple_scab/
│   ├── Apple___Black_rot/
│   └── ... (all 38 class directories)
└── PlantDoc/
    ├── train/
    └── test/
        ├── Apple Scab Leaf/
        ├── Apple leaf/
        └── ... (27 disease classes)
```

---

## 4. Environment Setup & Installation

### Step 1: Clone Repository
```bash
git clone https://github.com/Group12ITP/plant-disease-dl-comparison.git
cd plant-disease-dl-comparison
```

### Step 2: Create and Activate Virtual Environment
```bash
# Using conda
conda create -n plant-dl python=3.10 -y
conda activate plant-dl

# Or using venv (Windows PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

*Hardware Requirements:* An NVIDIA GPU supporting CUDA 12.0+ (e.g., NVIDIA Tesla T4, RTX 3060 or higher) with at least 8 GB VRAM is strongly recommended. Google Colab T4 GPU instances are fully supported.

---

## 5. Execution Pipeline (Replication Guide)

Follow this execution order to replicate all reported experimental results:

1. **Exploratory Data Analysis (`EDA.ipynb`):**
   Scans the raw image directory, audits data integrity, generates the 38-class distribution chart (`eda_class_distribution.png`), and computes the 35.24x imbalance ratio.
2. **Stratified Split & Custom CNN Training (`Custom CNN.ipynb`):**
   Executes the stratified 80/10/10 split, programmatically asserts zero data leakage, saves split manifests to `results/splits/`, and trains the from-scratch 4-block CNN.
3. **Transfer Learning Backbones (`ResNet.ipynb`, `Mobilenetv2.ipynb`, `EfficientNetB0.ipynb`):**
   Trains ResNet50, MobileNetV2, and EfficientNetB0 under identical conditions with frozen ImageNet backbones and customized classification heads.
4. **Out-of-Distribution Generalization (`Model Comparison.ipynb`):**
   Maps 27 PlantDoc classes to PlantVillage indices and benchmarks all 4 frozen models on real-world field imagery, quantifying the generalization gap.
5. **Few-Shot Domain Adaptation (`Fine-Tuning EfficientNetB0 Ablation.ipynb`):**
   Unfreezes the top 20 layers of EfficientNetB0 and fine-tunes with a reduced learning rate ($1 \times 10^{-4}$) using 5 field images per class.
6. **Multi-Class ROC-AUC Evaluation (`ROC-AUC.ipynb`):**
   Computes multiclass One-vs-Rest (OvR) Macro and Weighted ROC-AUC for all four models and renders comparative ROC curves (`roc_curves_plantvillage.png`).

---

## 6. Experimental Configurations & Hyperparameters

To ensure deterministic, fair, and reproducible comparison:
- **Global Random Seed:** `42` (applied uniformly across NumPy, TensorFlow, Python `random`, and Scikit-Learn).
- **Input Resolution:** $224 \times 224 \times 3$ (bilinear interpolation).
- **Batch Size:** 32.
- **Optimization Algorithm:** Adam ($\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-7}$).
- **Base Learning Rate:** $\eta = 1.0 \times 10^{-3}$ (initial); decayed via `ReduceLROnPlateau` ($\text{factor} = 0.5, \text{patience} = 2, \eta_{\min} = 10^{-6}$).
- **Regularization:** Spatial Dropout ($0.2 - 0.4$), Batch Normalization, and Early Stopping with weight restoration.
- **Data Augmentation (Train only):** Random horizontal/vertical flips, brightness ($\pm 12\%$), contrast ($[0.85, 1.15]$), and numeric clipping to $[0.0, 1.0]$.

---

## 7. Key Experimental Findings

### 7.1 In-Distribution Benchmark (PlantVillage Test Set, $N = 5,431$)

| Architecture | Total Params | Train Time | Test Acc (%) | Macro F1 | Weighted F1 | Macro ROC-AUC | Weighted ROC-AUC | Latency (ms) |
|---|---|---|---|---|---|---|---|---|
| **Custom CNN** | **285,030** | 26.28 min | **99.08%** | **0.9868** | **0.9908** | **0.99997** | **0.99997** | **1.89 ms** |
| **ResNet50** | 24,122,022 | 30.47 min | 98.60% | 0.9799 | 0.9859 | 0.99991 | 0.99993 | 6.52 ms |
| **MobileNetV2** | 2,595,686 | **16.35 min** | 96.13% | 0.9535 | 0.9613 | 0.99958 | 0.99961 | 6.05 ms |
| **EfficientNetB0** | 4,387,273 | 16.37 min | 98.36% | 0.9785 | 0.9835 | 0.99990 | 0.99993 | 5.70 ms |

### 7.2 Out-of-Distribution Generalization (PlantDoc Field Test Set, $N = 236$)

| Architecture | PlantVillage Acc (%) | PlantDoc OOD Acc (%) | Generalization Drop ($\Delta$) | PlantDoc Macro F1 |
|---|---|---|---|---|
| **Custom CNN** | 99.08% | 15.68% | **-83.40 pp** | 0.1045 |
| **ResNet50** | 98.60% | 22.46% | -76.14 pp | 0.1782 |
| **MobileNetV2** | 96.13% | 23.73% | -72.40 pp | **0.1946** |
| **EfficientNetB0** | 98.36% | **25.85%** | **-72.51 pp** | 0.1869 |

### 7.3 Few-Shot Domain Adaptation (EfficientNetB0 Ablation)
- **Baseline Zero-Shot Accuracy on PlantDoc:** 25.85%
- **Domain-Adapted Accuracy (5 images/class fine-tuning):** **36.27%**
- **Empirical Recovery:** **+10.42 percentage points**

---

## 8. Academic References

1. D. P. Hughes and M. Salathé, "An open access repository of images on plant health to enable the development of mobile disease diagnostics," *arXiv preprint arXiv:1511.08060*, 2015.
2. P. Singh, A. Verma, and A. Dosovitskiy, "PlantDoc: A Dataset for Collaborative Computer Vision Applications in Agriculture," in *Proc. 7th ACM IKDD CoDS and 25th COMAD*, 2020, pp. 249–256.
3. K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2016, pp. 770–778.
4. M. Sandler, A. Howard, M. Mengelong, A. Zhmoginov, and L.-C. Chen, "MobileNetV2: Inverted residuals and linear bottlenecks," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2018, pp. 4510–4520.
5. M. Tan and Q. V. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2019, pp. 6105–6114.
6. R. Geirhos, J. H. Jacobsen, C. Michaelis, R. S. Zemel, W. Brendel, M. Bethge, and F. A. Wichmann, "Shortcut learning in deep neural networks," *Nature Machine Intelligence*, vol. 2, no. 11, pp. 665–673, 2020.

---

## 9. Contributors

**Group ID:** Group 12  
**Module:** SE4050 – Deep Learning (SLIIT, 2026)  
- [Group Leader Name] - [Registration Number]
- [Member 2 Name] - [Registration Number]
- [Member 3 Name] - [Registration Number]
- [Member 4 Name] - [Registration Number]
