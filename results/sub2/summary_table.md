| Model | Mode | Trainable params | Epochs (best) | Train time (min) | Test time (s) | ms / image | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) | Parasitized recall (%) | AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ResNet-50 | Frozen (no fine-tuning) | 4,098 | 33 (25) | 8.4 | 3.77 | 0.955 | 93.41 | 93.43 | 93.41 | 93.41 | 92.6 | 0.9781 |
| ResNet-50 | Fine-tuned | 23.51 M | 30 (8) | 14.4 | 3.14 | 0.795 | 97.29 | 97.29 | 97.29 | 97.29 | 97.16 | 0.9964 |
| DenseNet-121 | Frozen (no fine-tuning) | 2,050 | 39 (31) | 9.7 | 4.55 | 1.151 | 92.43 | 92.45 | 92.43 | 92.43 | 91.28 | 0.9757 |
| DenseNet-121 | Fine-tuned | 6.96 M | 30 (19) | 16.4 | 3.29 | 0.834 | 97.7 | 97.7 | 97.69 | 97.69 | 97.31 | 0.9968 |
| MobileNetV2 | Frozen (no fine-tuning) | 2,562 | 30 (17) | 6.9 | 4.37 | 1.106 | 92.33 | 92.34 | 92.32 | 92.32 | 91.43 | 0.9737 |
| MobileNetV2 | Fine-tuned | 2.23 M | 30 (10) | 7.6 | 3.17 | 0.802 | 97.29 | 97.29 | 97.29 | 97.29 | 97.26 | 0.9962 |