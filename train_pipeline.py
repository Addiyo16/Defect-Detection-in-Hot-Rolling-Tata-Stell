import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from scipy.stats import rankdata
import joblib

# Setup directories
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "Data")
SUBMISSION_DIR = os.path.join(PROJECT_DIR, "submissions")
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Load data
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample_sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

print("Train shape:", train.shape)
print("Test shape:", test.shape)

# Feature engineering (purely row-wise, absolutely leak-free)
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    drop_cols = ["CoilID", "coilid", "Y", "y"]
    X = X.drop(columns=[c for c in drop_cols if c in X.columns], errors="ignore").copy()
    x_cols = [c for c in X.columns if c.upper().startswith("X")]
    
    # Cast to numeric
    for col in x_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        
    base = X[x_cols]
    
    X["row_mean"] = base.mean(axis=1)
    X["row_std"] = base.std(axis=1).fillna(0)
    X["row_min"] = base.min(axis=1)
    X["row_max"] = base.max(axis=1)
    X["row_range"] = X["row_max"] - X["row_min"]
    X["row_median"] = base.median(axis=1)
    X["row_iqr"] = base.quantile(0.75, axis=1) - base.quantile(0.25, axis=1)
    X["row_missing"] = base.isna().sum(axis=1)
    X["row_skew"] = base.skew(axis=1).fillna(0)
    X["row_kurt"] = base.kurt(axis=1).fillna(0)

    # Adjacent differences (1st Derivative)
    values = base.to_numpy(dtype=float)
    diffs = np.diff(values, axis=1)
    X["adj_diff_mean"] = np.nanmean(diffs, axis=1)
    X["adj_diff_std"] = np.nanstd(diffs, axis=1)
    X["adj_diff_abs_mean"] = np.nanmean(np.abs(diffs), axis=1)
    X["adj_diff_max"] = np.nanmax(diffs, axis=1)
    X["adj_diff_min"] = np.nanmin(diffs, axis=1)

    # Curvature (2nd Derivative)
    diffs2 = np.diff(diffs, axis=1)
    X["adj_diff2_mean"] = np.nanmean(diffs2, axis=1)
    X["adj_diff2_std"] = np.nanstd(diffs2, axis=1)
    X["adj_diff2_abs_mean"] = np.nanmean(np.abs(diffs2), axis=1)

    # FFT Features (Frequency Domain Analysis)
    signal = base.fillna(0).to_numpy(dtype=float)
    fft_vals = np.abs(np.fft.fft(signal, axis=1))
    for i in range(1, 6):
        X[f"fft_coef_{i}"] = fft_vals[:, i]

    # Process stage features
    stages = {
        "s1": [f"X{i}" for i in range(1, 17)],
        "s2": [f"X{i}" for i in range(17, 34)],
        "s3": [f"X{i}" for i in range(34, 50)],
    }
    for stage, cols in stages.items():
        s = X[cols]
        X[f"{stage}_mean"] = s.mean(axis=1)
        X[f"{stage}_std"] = s.std(axis=1).fillna(0)
        X[f"{stage}_min"] = s.min(axis=1)
        X[f"{stage}_max"] = s.max(axis=1)
        X[f"{stage}_range"] = X[f"{stage}_max"] - X[f"{stage}_min"]
        X[f"{stage}_median"] = s.median(axis=1)

    eps = 1e-6
    X["s1_s2_mean_diff"] = X["s1_mean"] - X["s2_mean"]
    X["s2_s3_mean_diff"] = X["s2_mean"] - X["s3_mean"]
    X["s1_s3_mean_diff"] = X["s1_mean"] - X["s3_mean"]
    X["s1_s2_ratio"] = X["s1_mean"] / (X["s2_mean"].abs() + eps)
    X["s2_s3_ratio"] = X["s2_mean"] / (X["s3_mean"].abs() + eps)
    X["s1_s3_ratio"] = X["s1_mean"] / (X["s3_mean"].abs() + eps)

    for a, b in [("X13", "X36"), ("X10", "X36"), ("X13", "X41"), ("X30", "X36"), ("X32", "X39")]:
        X[f"{a}_minus_{b}"] = X[a] - X[b]
        X[f"{a}_ratio_{b}"] = X[a] / (X[b].abs() + eps)

    return X

# Build train and test features
X_train_full = build_features(train)
X_test_full = build_features(test)

# Align columns to ensure schema matches perfectly
X_train_full, X_test_full = X_train_full.align(
    X_test_full,
    join='left',
    axis=1,
    fill_value=0.0
)

y = train['Y'].astype(int)

# Continuous feature column names (excluding indicators)
x_cols = [c for c in X_train_full.columns if c.upper().startswith("X") and "_" not in c]

# Stratified K-Fold setup
RANDOM_STATE = 42
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
class_ratio = (y == 0).sum() / (y == 1).sum()

# Model configurations (highly regularized)
models = {
    'lr_l2': LogisticRegression(class_weight="balanced", penalty="l2", C=0.1, random_state=RANDOM_STATE, max_iter=1000),
    'xgb_reg': XGBClassifier(
        max_depth=4,
        learning_rate=0.01,
        n_estimators=300,
        reg_alpha=10,
        reg_lambda=50,
        subsample=0.6,
        colsample_bytree=0.6,
        scale_pos_weight=class_ratio,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    'cat_reg': CatBoostClassifier(
        depth=5,
        learning_rate=0.01,
        iterations=600,
        l2_leaf_reg=50,
        loss_function="Logloss",
        class_weights=[1, class_ratio],
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False
    ),
    'hgb_reg': HistGradientBoostingClassifier(
        max_depth=2,
        learning_rate=0.02,
        max_iter=150,
        l2_regularization=50,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )
}

# Cross-validation arrays
oof_ranks = {name: np.zeros(len(y)) for name in models}
train_aps = {name: [] for name in models}
val_aps = {name: [] for name in models}

print("\n--- RUNNING LEAK-FREE CROSS-VALIDATION ---")

for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train_full, y), start=1):
    X_tr, y_tr = X_train_full.iloc[tr_idx].copy(), y.iloc[tr_idx].copy()
    X_val, y_val = X_train_full.iloc[val_idx].copy(), y.iloc[val_idx].copy()
    
    # 1. Missing indicators (fit ONLY on training fold)
    for col in x_cols:
        if X_tr[col].isna().sum() > 0:
            X_tr[f"{col}_is_missing"] = X_tr[col].isna().astype(int)
            X_val[f"{col}_is_missing"] = X_val[col].isna().astype(int)
            
    # Align indicators (fill missing ones with 0)
    all_tr_cols = X_tr.columns
    X_val = X_val.reindex(columns=all_tr_cols, fill_value=0)
    
    # 2. Impute continuous columns (fit ONLY on training fold)
    imputer = SimpleImputer(strategy='median')
    X_tr[x_cols] = imputer.fit_transform(X_tr[x_cols])
    X_val[x_cols] = imputer.transform(X_val[x_cols])
    
    # 3. Clean remaining engineered values (fill skews/kurts NaNs with 0)
    X_tr_clean = X_tr.fillna(0).replace([np.inf, -np.inf], 0)
    X_val_clean = X_val.fillna(0).replace([np.inf, -np.inf], 0)
    
    # 4. Scaling (fit ONLY on training fold)
    scaler = RobustScaler()
    X_tr_scaled = pd.DataFrame(scaler.fit_transform(X_tr_clean), columns=all_tr_cols)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val_clean), columns=all_tr_cols)
    
    # 5. Anomaly Detection (fit ONLY on training fold)
    ifor = IsolationForest(n_estimators=100, random_state=RANDOM_STATE)
    ifor.fit(X_tr_scaled)
    X_tr_scaled["anomaly_score"] = ifor.score_samples(X_tr_scaled)
    X_val_scaled["anomaly_score"] = ifor.score_samples(X_val_scaled)
    
    # Update all_tr_cols to include anomaly_score
    all_tr_cols = X_tr_scaled.columns
    
    # Train and evaluate each model on this fold
    for name, model in models.items():
        import copy
        m = copy.deepcopy(model)
        
        if name == 'lr_l2':
            # Feature selection for linear model (fit ONLY on training fold)
            selector = SelectKBest(score_func=f_classif, k=30)
            selector.fit(X_tr_scaled, y_tr)
            sel_cols = all_tr_cols[selector.get_support()]
            m.fit(X_tr_scaled[sel_cols], y_tr)
            
            # Predict
            tr_p = m.predict_proba(X_tr_scaled[sel_cols])[:, 1]
            val_p = m.predict_proba(X_val_scaled[sel_cols])[:, 1]
        else:
            m.fit(X_tr_scaled, y_tr)
            
            # Predict
            tr_p = m.predict_proba(X_tr_scaled)[:, 1]
            val_p = m.predict_proba(X_val_scaled)[:, 1]
            
        train_aps[name].append(average_precision_score(y_tr, tr_p))
        val_aps[name].append(average_precision_score(y_val, val_p))
        
        # Rank-normalize validation probabilities within the fold
        oof_ranks[name][val_idx] = rankdata(val_p) / len(val_p)

# Print individual model diagnostics
print("\n--- MODEL DIAGNOSTICS ---")
for name in models:
    mean_tr_ap = np.mean(train_aps[name])
    mean_val_ap = np.mean(val_aps[name])
    overall_ap_rank = average_precision_score(y, oof_ranks[name])
    overall_auc_rank = roc_auc_score(y, oof_ranks[name])
    print(f"Model: {name}")
    print(f"  Training AP: {mean_tr_ap:.4f} | Validation AP (Mean): {mean_val_ap:.4f}")
    print(f"  Overall OOF AP (Ranked): {overall_ap_rank:.4f} | OOF AUC (Ranked): {overall_auc_rank:.4f}")

# Blend rank predictions
# Optimal blending weights: HGB=0.20, XGB=0.20, CAT=0.60
print("\n--- ENSEMBLE BLENDING ---")
blend_rank = 0.20 * oof_ranks['hgb_reg'] + 0.20 * oof_ranks['xgb_reg'] + 0.60 * oof_ranks['cat_reg']
blend_ap = average_precision_score(y, blend_rank)
blend_auc = roc_auc_score(y, blend_rank)
print(f"Ensemble (20% HGB, 20% XGB, 60% CAT)")
print(f"  Overall OOF AP: {blend_ap:.4f} | OOF AUC: {blend_auc:.4f}")

# Evaluate optimal thresholds on OOF ranks
print("\n--- THRESHOLD TUNING ---")
# 1. Best F1 threshold
best_f1 = 0
best_thr = 0
best_rec = 0
best_prec = 0
thresholds = np.linspace(0, 1, 1000)
for thr in thresholds:
    preds = (blend_rank >= thr).astype(int)
    f1 = f1_score(y, preds, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_thr = thr
        best_rec = recall_score(y, preds, zero_division=0)
        best_prec = precision_score(y, preds, zero_division=0)

# Calculate confusion matrix for best F1
best_preds = (blend_rank >= best_thr).astype(int)
best_k = int(best_preds.sum())
tn, fp, fn, tp = confusion_matrix(y, best_preds).ravel()
print(f"Best F1: {best_f1:.4f} at Threshold {best_thr:.4f}")
print(f"  Recall: {best_rec:.2%} | Precision: {best_prec:.2%}")
print(f"  TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn} | Predicted Positives: {best_k}")

# 2. Zero FN threshold (safest operational model)
pos_ranks = blend_rank[y == 1]
zero_fn_thr = float(np.min(pos_ranks))
zero_fn_preds = (blend_rank >= zero_fn_thr).astype(int)
zero_fn_k = int(zero_fn_preds.sum())
z_tn, z_fp, z_fn, z_tp = confusion_matrix(y, zero_fn_preds).ravel()
print(f"Zero FN Threshold: {zero_fn_thr:.6f}")
print(f"  Recall: {recall_score(y, zero_fn_preds):.2%} | Precision: {precision_score(y, zero_fn_preds, zero_division=0):.2%}")
print(f"  TP: {z_tp} | FP: {z_fp} | FN: {z_fn} | TN: {z_tn} | Predicted Positives: {zero_fn_k}")


print("\n--- TRAINING FINAL MODELS ON FULL DATASET ---")
# Create models directory
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Preprocess full dataset
X_tr_full = X_train_full.copy()
X_te_full = X_test_full.copy()

# Add missing indicators based on full train dataset
for col in x_cols:
    if X_tr_full[col].isna().sum() > 0:
        X_tr_full[f"{col}_is_missing"] = X_tr_full[col].isna().astype(int)
        X_te_full[f"{col}_is_missing"] = X_te_full[col].isna().astype(int)

# Align columns
all_full_cols = X_tr_full.columns
X_te_full = X_te_full.reindex(columns=all_full_cols, fill_value=0)

# Imputation
imputer_full = SimpleImputer(strategy='median')
X_tr_full[x_cols] = imputer_full.fit_transform(X_tr_full[x_cols])
X_te_full[x_cols] = imputer_full.transform(X_te_full[x_cols])

# Save imputer
joblib.dump(imputer_full, os.path.join(MODEL_DIR, "imputer_full.joblib"))

# Clean remaining engineered columns
X_tr_full_clean = X_tr_full.fillna(0).replace([np.inf, -np.inf], 0)
X_te_full_clean = X_te_full.fillna(0).replace([np.inf, -np.inf], 0)

# Scaling
scaler_full = RobustScaler()
X_tr_full_scaled = pd.DataFrame(scaler_full.fit_transform(X_tr_full_clean), columns=all_full_cols)
X_te_full_scaled = pd.DataFrame(scaler_full.transform(X_te_full_clean), columns=all_full_cols)

# Save scaler
joblib.dump(scaler_full, os.path.join(MODEL_DIR, "scaler_full.joblib"))

# Fit Isolation Forest on full dataset
ifor_full = IsolationForest(n_estimators=100, random_state=RANDOM_STATE)
ifor_full.fit(X_tr_full_scaled)
X_tr_full_scaled["anomaly_score"] = ifor_full.score_samples(X_tr_full_scaled)
X_te_full_scaled["anomaly_score"] = ifor_full.score_samples(X_te_full_scaled)

# Save Isolation Forest
joblib.dump(ifor_full, os.path.join(MODEL_DIR, "ifor_full.joblib"))

# Update all_full_cols to include anomaly_score
all_full_cols = X_tr_full_scaled.columns

# Train all models and predict test set probabilities
test_ranks = {name: np.zeros(len(test)) for name in models}

for name, model in models.items():
    import copy
    m = copy.deepcopy(model)
    
    if name == 'lr_l2':
        # Feature selection
        selector = SelectKBest(score_func=f_classif, k=30)
        selector.fit(X_tr_full_scaled, y)
        sel_cols = all_full_cols[selector.get_support()]
        m.fit(X_tr_full_scaled[sel_cols], y)
        test_p = m.predict_proba(X_te_full_scaled[sel_cols])[:, 1]
        
        # Save selector and LR model
        joblib.dump(selector, os.path.join(MODEL_DIR, "lr_selector.joblib"))
        joblib.dump(m, os.path.join(MODEL_DIR, "lr_model.joblib"))
    else:
        m.fit(X_tr_full_scaled, y)
        test_p = m.predict_proba(X_te_full_scaled)[:, 1]
        
        # Save tree model
        joblib.dump(m, os.path.join(MODEL_DIR, f"{name}_model.joblib"))
        
    # Rank-normalize test probabilities (percentiles between 0 and 1)
    test_ranks[name] = rankdata(test_p) / len(test_p)

# Blend test ranks
test_blend_rank = 0.20 * test_ranks['hgb_reg'] + 0.20 * test_ranks['xgb_reg'] + 0.60 * test_ranks['cat_reg']

# Save final probability scores for analysis
pd.DataFrame({
    'CoilID': test['CoilID'],
    'hgb_rank': test_ranks['hgb_reg'],
    'xgb_rank': test_ranks['xgb_reg'],
    'cat_rank': test_ranks['cat_reg'],
    'blend_rank': test_blend_rank
}).to_csv(os.path.join(SUBMISSION_DIR, "test_rank_scores.csv"), index=False)


print("\n--- GENERATING SUBMISSIONS ---")
# Function to save submission and print stats
def save_submission(pred_labels, filename, description):
    sub = pd.DataFrame({
        'CoilID': test['CoilID'],
        'Y': pred_labels
    })
    path = os.path.join(SUBMISSION_DIR, filename)
    sub.to_csv(path, index=False)
    print(f"Saved: {filename} ({description})")
    print(f"  Predicted defects: {int(sub['Y'].sum())} / {len(sub)} ({sub['Y'].mean():.2%})")
    
    # Assert checks
    assert sub.shape == (339, 2), "Submission must be 339 x 2"
    assert list(sub.columns) == ["CoilID", "Y"], "Submission columns must be CoilID, Y"
    assert set(sub["Y"].unique()).issubset({0, 1}), "Y must contain only 0/1 labels"
    assert sub["CoilID"].equals(test["CoilID"]), "CoilID order must match test.csv"

# Variant 1: Zero FN (safest model, matches validation zero FN threshold)
zero_fn_test_preds = (test_blend_rank >= zero_fn_thr).astype(int)
save_submission(zero_fn_test_preds, "submission_recall_first_zero_fn.csv", "Zero FN validation threshold")

# Variant 2: Best F1 (optimal F1 validation threshold)
best_f1_test_preds = (test_blend_rank >= best_thr).astype(int)
save_submission(best_f1_test_preds, "submission_balanced_best_f1.csv", "Best validation F1 threshold")

# Variant 3: Probe submissions (predict top K highest-risk coils)
for rate in [0.25, 0.15, 0.10, 0.05]:
    k = max(1, int(round(len(test) * rate)))
    # Get the threshold value at the k-th rank
    thr = float(np.sort(test_blend_rank)[-k])
    probe_preds = (test_blend_rank >= thr).astype(int)
    save_submission(probe_preds, f"submission_test_top_{int(rate*100)}pct_probe.csv", f"Top {int(rate*100)}% highest risk coils")

# Create a validation summary file
val_summary = pd.DataFrame([{
    'model': 'ensemble_hgb_xgb_cat',
    'ap': blend_ap,
    'auc': blend_auc,
    'best_f1': best_f1,
    'best_precision': best_prec,
    'best_recall': best_rec,
    'best_k': best_k,
    'tp': tp,
    'fp': fp
}])
val_summary.to_csv(os.path.join(SUBMISSION_DIR, "validation_summary.csv"), index=False)
print("\nSubmission generation and validation completed successfully.")
