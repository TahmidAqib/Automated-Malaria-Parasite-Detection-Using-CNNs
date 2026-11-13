# Automated Malaria Parasite Detection in Thin Blood Smears Using CNNs

**CSE366 Artificial Intelligence — Group 6**

Convolutional neural networks that decide, from a microscope image of a single red blood cell, whether the cell is infected with the malaria parasite (*Plasmodium*). The project compares three ImageNet-pretrained CNNs (frozen and fine-tuned) with a CNN we designed and trained from scratch, on a leakage-safe split of the NIH Malaria Cell Images dataset.

**Best model:** fine-tuned DenseNet-121, **97.70%** test accuracy, **97.31%** Parasitized recall, AUC **0.997**.
**Our custom CNN:** **97.14%** test accuracy with only **1.21 M** parameters (5.8× fewer than DenseNet-121).

| Name | Student ID |
|---|---|
| Tahmid Aqib | 2024-1-60-225 |
| Rejoanur Rahman | 2024-1-60-245 |
| Mahmudur Rahaman Pial | 2024-1-60-177 |
| Sayem Siddiq | 2024-1-60-298 |

---

## Notebooks

| Task | Notebook | Contents |
|---|---|---|
| 1 | [`Submission1_Dataset_EDA_RelatedWork.ipynb`](Submission1_Dataset_EDA_RelatedWork.ipynb) | Dataset documentation, exploratory data analysis, leakage-safe split, related-work table |
| 2 | [`Submission2_Pretrained_Models.ipynb`](Submission2_Pretrained_Models.ipynb) | ResNet-50, DenseNet-121, MobileNetV2 — each frozen and fully fine-tuned |
| 3 | [`Submission3_Custom_CNN.ipynb`](Submission3_Custom_CNN.ipynb) | Custom CNN trained from scratch, error analysis, clinical threshold study, comparison with Task 2 |

All notebooks are committed with their outputs, so every plot and table is visible on GitHub without re-running.

---

## Dataset

| | |
|---|---|
| **Name** | Malaria Cell Images Dataset (NIH / U.S. National Library of Medicine, 2018) |
| **Download** | <https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria> |
| **Task** | Binary classification: `Parasitized` vs `Uninfected` |
| **Images** | 27,558 RGB PNGs — **13,779 per class** (perfectly balanced) |
| **Image size** | Variable, 46–394 × 40–385 px, median 130×130 px |
| **Origin** | Giemsa-stained thin blood smears from Chittagong, Bangladesh, photographed with a smartphone through a microscope; 1,408 distinct smear photographs |

The images are **not** included in this repository (see [How to run](#how-to-run)).

### Leakage-safe split

Many cells were cut from the same smear photograph and share its lighting, focus and stain. A random split would place near-identical cells in both training and test sets and inflate scores. We therefore split **by source photograph** using `StratifiedGroupKFold` (seed 42):

| Subset | Images | Share |
|---|---|---|
| Train | 19,661 | 71.3% |
| Validation | 3,949 | 14.3% |
| Test | 3,948 | 14.3% |

Every notebook uses this exact split, and each model touches the test set only once.

---

## Methods

**Preprocessing:** pad each cell to a square with black (keeps cells round), resize to 224×224 (pretrained models) or 128×128 (custom CNN), normalise with ImageNet statistics (pretrained) or our training-set statistics (custom CNN).

**Augmentation (training only):** random 0–360° rotation, horizontal and vertical flips, ±5% shift and zoom, mild colour jitter. No class weights (balanced data).

**Task 2 — Transfer learning.** Each model is trained twice from fresh ImageNet weights:
- *Frozen:* only a new head (Dropout → Linear → 2 classes) learns; backbone weights and BatchNorm statistics are fixed.
- *Fine-tuned:* all layers learn (backbone lr 1e-4, head lr 1e-3).

30–40 epochs, AdamW, ReduceLROnPlateau, early stopping after epoch 30, best-validation checkpoint restored before testing.

**Task 3 — Custom CNN.** Four blocks of `[Conv3×3 → BatchNorm → ReLU] ×2 → MaxPool`, channels 32 → 64 → 128 → 256, then global average pooling → Dropout 0.5 → FC 256→128 → ReLU → Dropout 0.3 → FC 128→2. 1,206,370 parameters, Kaiming initialisation. AdamW (lr 1e-3), ReduceLROnPlateau, early stopping (patience 8, minimum 5 epochs, maximum 100).

**Shared settings:** batch size 64, mixed precision (FP16), channels-last memory layout, seed 42.

---

## Results (test set: 3,948 images, 1,972 infected)

| Model | Mode | Accuracy | Macro F1 | Parasitized recall | Missed infections | AUC | Training time |
|---|---|---|---|---|---|---|---|
| **DenseNet-121** | **Fine-tuned** | **97.70%** | **97.69%** | **97.31%** | **53** | **0.9968** | 16.4 min |
| ResNet-50 | Fine-tuned | 97.29% | 97.29% | 97.16% | 56 | 0.9964 | 14.4 min |
| MobileNetV2 | Fine-tuned | 97.29% | 97.29% | 97.26% | 54 | 0.9962 | 7.6 min |
| Custom CNN | From scratch | 97.14% | 97.14% | 96.86% | 62 | 0.9953 | 6.7 min |
| ResNet-50 | Frozen | 93.41% | 93.41% | 92.60% | 146 | 0.9781 | 8.4 min |
| DenseNet-121 | Frozen | 92.43% | 92.43% | 91.28% | 172 | 0.9757 | 9.7 min |
| MobileNetV2 | Frozen | 92.33% | 92.32% | 91.43% | 169 | 0.9737 | 6.9 min |

**Key findings**
- **Fine-tuning is essential:** +3.9 to +5.3 accuracy points over frozen ImageNet features.
- **A small custom CNN nearly matches large pretrained models:** within 0.56 points of DenseNet-121, with 5.8× fewer parameters, less than half the training time and the fastest inference (0.54 ms/image).
- **No overfitting at the selected epochs:** train–validation gaps between −2.74 and +0.30 points; the custom CNN stopped early at epoch 63 (best epoch 55).
- **Parasitized recall is the critical metric** (a missed infection is far worse than a false alarm). Lowering the custom CNN's decision threshold, chosen on validation data, raises recall to 98.43% and halves missed infections (62 → 31), at the cost of more false alarms (51 → 158).

---

## How to run

Tested on Fedora Linux 45, Python 3.15, PyTorch 2.15 (CUDA), NVIDIA RTX 4070 (12 GB), 16 GB RAM.

```bash
git clone https://github.com/TahmidAqib/Automated-Malaria-Parasite-Detection-Using-CNNs.git
cd Automated-Malaria-Parasite-Detection-Using-CNNs

python3 -m pip install --user jupyter notebook torch torchvision numpy pandas \
    matplotlib pillow scikit-learn tqdm kagglehub
python3 -m jupyter notebook
```

**Data:** extract the Kaggle zip into `./data/`, or let the notebooks download it automatically with `kagglehub` (needs a Kaggle API token).

**Run the notebooks in order** (Kernel → Restart & Run All):
1. **Task 1** creates `splits/splits.csv` and `splits/norm_stats.json`, which the other notebooks need.
2. **Task 2** writes `malaria_data.py` (the shared `Dataset` class) and trains six models (~1 hour on an RTX 4070).
3. **Task 3** imports `malaria_data.py` and trains the custom CNN (~7 minutes).

> **Note for Python 3.14+ on Linux:** DataLoader workers start with `forkserver` and cannot import classes defined inside a notebook cell. That is why the `Dataset` class lives in `malaria_data.py`, written by the Task 2 notebook. If you see `AttributeError: module '__main__' has no attribute 'MalariaCellDataset'`, make sure that file sits next to the notebooks. If system RAM runs low, reduce `NUM_WORKERS` in the configuration cell.

---

## References

1. Liang et al., "CNN-based image analysis for malaria diagnosis," IEEE BIBM, 2016.
2. Rajaraman et al., "Pre-trained convolutional neural networks as feature extractors toward improved malaria parasite detection in thin blood smear images," *PeerJ* 6:e4568, 2018. <https://doi.org/10.7717/peerj.4568>
3. Rajaraman, Jaeger & Antani, "Performance evaluation of deep neural ensembles toward malaria parasite detection in thin-blood smear images," *PeerJ* 7:e6977, 2019. <https://doi.org/10.7717/peerj.6977>
4. Fuhad et al., "Deep learning based automatic malaria parasite detection from blood smear and its smartphone based application," *Diagnostics* 10(5):329, 2020. <https://doi.org/10.3390/diagnostics10050329>
5. Ghosh et al., "A convolutional neural network with attention mechanism-based malaria detection from blood smear images," *Int. J. Advances in Applied Sciences*, 2025. <https://ijaas.iaescore.com/index.php/IJAAS/article/view/21844>

**Acknowledgements:** dataset by the U.S. National Library of Medicine (Lister Hill National Center for Biomedical Communications, NIH); pretrained weights from torchvision. Course: CSE366 Artificial Intelligence, East West University.
