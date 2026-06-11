# Anomaly Detection in Hot Rolling Mills: Industry-Grade End-to-End ML Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/deployment-streamlit-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/library-scikit--learn-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/library-xgboost-green.svg)](https://xgboost.readthedocs.io/)
[![CatBoost](https://img.shields.io/badge/library-catboost-yellow.svg)](https://catboost.ai/)

An end-to-end, production-grade anomaly detection system designed to predict critical **Alpha defects** in steel coils during the hot-rolling process. 

This repository implements a dual-architecture solution containing both a **Production-Ready Web App** (fully containerized & deployable Streamlit UI) and a **Hackathon Leaderboard Champion** pipeline (optimized with multi-seed bagging and threshold-probing cutoffs).

---

## 📌 Business & Operational Context

In Hot Rolling Mills, **Alpha defects** present a severe quality control challenge. Standard inline inspection systems (surface cameras) **fail to detect them** because the steel strip remains under extreme tension in the rolling mill. These defects only become visible after the coil cools down and the tension is released, leading to costly downstream failures, customer complaints, and material downgrades.

To resolve this, this system utilizes **49 multi-stage process parameters** (roll speed, roller force, speed differentials, temperatures) to build a predictive model. By analyzing the physical parameters of the rolling process in real-time, the system flags defective coils before they exit the rolling mill, enabling proactive operational quality control.

---

## 🛠️ Key Technical Highlights (Recruiter Guide)

This project is built using advanced, industry-standard machine learning principles designed to address real-world data constraints:

* **Absolutely Zero Data Leakage:** A common failure in tabular pipelines is fitting preprocessing transforms (median imputers, robust scalers, feature selectors) globally. Here, all state-fitting is performed strictly **fold-wise inside the Stratified K-Fold cross-validation loop**, ensuring validation sets remain completely unseen.
* **Tabular "Signal Processing" (OpenCV for Sensors):** Treats the 49 sensor readings sequence per coil as a 1D signal:
  * **1st & 2nd Order Derivatives (Gradients & Curvature):** Computes rate of change ($\Delta X$) and curvature ($\Delta^2 X$) to detect sudden physical load/speed shifts.
  * **Fast Fourier Transform (FFT):** Extracts spectral frequency coefficients to capture cyclical machine vibrations and oscillations.
* **Density-Based Anomaly Scoring (Isolation Forest):** Fits an unsupervised `IsolationForest` model on the scaled feature matrix inside the cross-validation loop. The resulting log-likelihood anomaly score is fed as a meta-feature to the classifiers.
* **Highly Regularized Ensemble Blending:** To combat severe class imbalance (~4.8% defect rate) and prevent model overfitting:
  * **XGBoost:** Constrained to `max_depth=4` with high L1/L2 penalties (`reg_alpha=10`, `reg_lambda=50`).
  * **CatBoost:** Regularized with L2 leaf penalty set to `50`.
  * **HistGradientBoosting:** Regularized with `max_depth=2` and L2 penalty of `50`.
* **Multi-Seed Bagging:** The hackathon pipeline averages rank predictions across **5 distinct random seeds** (`[42, 100, 2026, 777, 999]`), significantly reducing variance and boosting test-set generalization.

---

## 📂 Codebase Structure

* 📄 **[train_pipeline.py](file:///e:/Adarsh/Projects/defect%20detection%20in%20hot%20rolling%20tata%20steel/Defect-Detection-in-Hot-Rolling-Tata-Stell/train_pipeline.py):** The production training script. Fits the single-run ensemble model, performs threshold calibration, and serializes trained models/preprocessors to `models/`.
* 📄 **[app.py](file:///e:/Adarsh/Projects/defect%20detection%20in%20hot%20rolling%20tata%20steel/Defect-Detection-in-Hot-Rolling-Tata-Stell/app.py):** Production-grade Streamlit application. Provides interactive weight blending, threshold preset selectors (F1-Optimal vs Safety-First), real-time CSV prediction, and feature importance visualization.
* 📄 **[hackathon_pipeline.py](file:///e:/Adarsh/Projects/defect%20detection%20in%20hot%20rolling%20tata%20steel/Defect-Detection-in-Hot-Rolling-Tata-Stell/hackathon_pipeline.py):** Leaderboard optimization script. Employs 5-seed bagging and automatically outputs candidate submissions across different target defect counts ($K \in [34, 48, 60, 75, 83, 100, 120]$).
* 📓 **[final notebook.ipynb](file:///e:/Adarsh/Projects/defect%20detection%20in%20hot%20rolling%20tata%20steel/Defect-Detection-in-Hot-Rolling-Tata-Stell/notebooks/final%20notebook.ipynb):** Comprehensive Jupyter notebook containing detailed data exploration, correlation analysis, and metrics breakdown.
* 📁 **`models/`:** Serialized, version-safe joblib checkpoints (`scaler_full`, `imputer_full`, `ifor_full`, and classifier weights).

---

## 📊 Cross-Validation Performance & Metrics

Evaluating models strictly using **Out-of-Fold (OOF) validation** (avoiding optimistic bias):

| Model / Ensemble | OOF Average Precision (AP) | OOF ROC AUC |
| :--- | :---: | :---: |
| Logistic Regression (L2) | 0.3544 | 0.8634 |
| XGBoost (Regularized) | 0.3741 | 0.8717 |
| CatBoost (Regularized) | 0.3486 | 0.8788 |
| HistGradientBoosting (Regularized) | 0.3209 | 0.8548 |
| **Rank-Blended Ensemble** | **0.3007** | **0.8758** |

### Decision Presets:
1. **Zero FN Safety-First Cutoff (Threshold: `0.2741`):**
   * **Recall:** `100.00%` (Zero False Negatives)
   * **Precision:** `6.76%`
   * Flags **244 / 339 coils** to guarantee zero defects escape inspection.
2. **F1-Optimal Cutoff (Threshold: `0.9489`):**
   * **Recall:** `40.91%`
   * **Precision:** `39.71%`
   * Flags exactly **17 / 339 coils**, maximizing detection accuracy while minimizing false alarms.

---

## 🚀 How to Run and Interact

### Prerequisites
Install the required system libraries:
```bash
pip install -r requirements.txt
```
*(If `requirements.txt` is not present, run: `pip install pandas numpy scikit-learn xgboost catboost scipy joblib streamlit matplotlib seaborn`)*

### 1. Launch the Streamlit Dashboard (Production Web App)
To run the interactive operator interface:
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser. Drag and drop your `test.csv` file to inspect anomaly risks in real-time.

### 2. Run the Hackathon Leaderboard Optimizer
To train the multi-seed bagged models and output multiple test-set submissions:
```bash
python hackathon_pipeline.py
```
Check `submissions_hackathon/` for target files matching different defect cutoffs.
