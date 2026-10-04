# Breast Cancer Classification

**CSI 5170/4170 — Machine Learning, Homework 1**  
**Author:** Muhammad Rehan Afzal, Oakland University

This project compares **Logistic Regression** and **Decision Tree** classifiers implemented with **scikit-learn** and independently **from scratch using NumPy**.

## Dataset and Workflow

- **Dataset:** Breast Cancer Wisconsin (Diagnostic), loaded through scikit-learn.
- **Size:** 569 samples and 30 numerical features.
- **Labels:** `0 = malignant`, `1 = benign`. Malignant is the positive class for evaluation.
- **Stratified split:** 398 training, 85 validation and 86 test samples (approximately 70/15/15).
- **EDA:** Data-quality checks, class distribution, feature scales, skewness, outliers, correlations and class-wise comparisons.
- Logistic Regression uses training-only standardization; Decision Trees use unscaled features. No samples or features were removed.
- Hyperparameters were selected using validation F1: **LR `C=0.1`**; **tree `max_depth=3`, `min_samples_leaf=5`**.

## Final Test Results

All models were evaluated on the same 86 test samples.

| Implementation | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression — Library | 0.9767 | 1.0000 | 0.9375 | 0.9677 | 0.9936 |
| Logistic Regression — Scratch | 0.9767 | 1.0000 | 0.9375 | 0.9677 | 0.9936 |
| Decision Tree — Library | 0.8953 | 0.8966 | 0.8125 | 0.8525 | 0.9010 |
| Decision Tree — Scratch | 0.8953 | 0.8966 | 0.8125 | 0.8525 | 0.9034 |

Confusion matrices are included in the notebook. Each Logistic Regression model missed **2 malignant samples** with **0 false positives**; each tree missed **6 malignant samples** with **3 false positives**. The tree AUC difference reflects different probability rankings.

## Run the Notebook

Open `breast cancer detection.ipynb` in [Google Colab](https://colab.research.google.com/drive/1Pt2wd5A05DSZbWxUEwZ7ASmAxaGTw7IV?usp=sharing) or Jupyter, then run all cells from top to bottom.

For a local environment, install:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn Jinja2
```

## Limitations

Results describe one dataset and one test split. No cross-validation or external-data evaluation was performed. This is an educational classification project; the results do not establish clinical suitability.
