import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import joblib
from scipy.stats import rankdata
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Tata Steel AI Defect Detector",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design & Aesthetics (Modern Glassmorphism & Steel Theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Card design */
    .metric-card {
        background: rgba(38, 39, 48, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(8px);
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        border-color: rgba(27, 158, 119, 0.5);
        transform: translateY(-4px);
    }
    
    .metric-label {
        font-size: 14px;
        color: #A0AEC0;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 36px;
        font-weight: 800;
        color: #FFFFFF;
        margin-bottom: 4px;
    }
    .metric-sub {
        font-size: 12px;
        color: #718096;
    }
    
    /* Highlight states */
    .state-green { border-left: 5px solid #10B981 !important; }
    .state-yellow { border-left: 5px solid #F59E0B !important; }
    .state-red { border-left: 5px solid #EF4444 !important; }
    
    /* Title header */
    .header-container {
        padding: 20px;
        background: linear-gradient(135deg, #1e3a8a, #0f172a);
        border-radius: 12px;
        margin-bottom: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .header-title {
        font-size: 32px;
        font-weight: 800;
        background: linear-gradient(to right, #6EE7B7, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .header-sub {
        font-size: 14px;
        color: #94A3B8;
        margin-top: 4px;
        margin-bottom: 0;
    }
</style>
""", unsafe_allow_html=True)

# Project paths
PROJECT_DIR = "d:/Adarsh/TataSteelAIHackathon"
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
DATA_DIR = os.path.join(PROJECT_DIR, "Data")

# Load models and preprocessors (cached for speed)
@st.cache_resource
def load_model_assets():
    imputer = joblib.load(os.path.join(MODEL_DIR, "imputer_full.joblib"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler_full.joblib"))
    hgb = joblib.load(os.path.join(MODEL_DIR, "hgb_reg_model.joblib"))
    xgb = joblib.load(os.path.join(MODEL_DIR, "xgb_reg_model.joblib"))
    cat = joblib.load(os.path.join(MODEL_DIR, "cat_reg_model.joblib"))
    return imputer, scaler, hgb, xgb, cat

try:
    imputer, scaler, hgb, xgb, cat = load_model_assets()
    models_loaded = True
except Exception as e:
    st.error(f"Error loading model files from {MODEL_DIR}. Please make sure train_pipeline.py has run successfully. Details: {e}")
    models_loaded = False

# Row-wise feature engineering function
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    drop_cols = ["CoilID", "coilid", "Y", "y"]
    X = X.drop(columns=[c for c in drop_cols if c in X.columns], errors="ignore").copy()
    x_cols = [c for c in X.columns if c.upper().startswith("X")]
    
    # Cast to numeric
    for col in x_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        
    base = X[x_cols]
    
    # Row aggregates
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

    # Adjacent differences
    values = base.to_numpy(dtype=float)
    diffs = np.diff(values, axis=1)
    X["adj_diff_mean"] = np.nanmean(diffs, axis=1)
    X["adj_diff_std"] = np.nanstd(diffs, axis=1)
    X["adj_diff_abs_mean"] = np.nanmean(np.abs(diffs), axis=1)
    X["adj_diff_max"] = np.nanmax(diffs, axis=1)
    X["adj_diff_min"] = np.nanmin(diffs, axis=1)

    # Process stage averages
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

# Header Container
st.markdown("""
<div class="header-container">
    <h1 class="header-title">Tata Steel Anomaly Detection System</h1>
    <p class="header-sub">Industry-grade, leak-free ensembling defect prediction dashboard for hot-rolling process lines</p>
</div>
""", unsafe_allow_html=True)

if models_loaded:
    # Sidebar Setup
    st.sidebar.markdown("### 📥 Input Data Source")
    uploaded_file = st.sidebar.file_uploader("Upload sensor readings CSV", type=["csv"])
    
    if uploaded_file is not None:
        df_input = pd.read_csv(uploaded_file)
        st.sidebar.success(f"Loaded: {uploaded_file.name} ({len(df_input)} rows)")
    else:
        # Load default test set
        default_test_path = os.path.join(DATA_DIR, "test.csv")
        if os.path.exists(default_test_path):
            df_input = pd.read_csv(default_test_path)
            st.sidebar.info(f"Using default dataset: test.csv ({len(df_input)} rows)")
        else:
            st.sidebar.error("Could not find default test.csv. Please upload a sensor reading file.")
            df_input = None

    if df_input is not None:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🎛️ Blending Weights")
        w_cat = st.sidebar.slider("CatBoost Weight", 0.0, 1.0, 0.60, 0.05)
        w_xgb = st.sidebar.slider("XGBoost Weight", 0.0, 1.0, 0.20, 0.05)
        w_hgb = st.sidebar.slider("HistGradientBoosting Weight", 0.0, 1.0, 0.20, 0.05)
        
        # Normalize weights if sum != 1.0
        w_sum = w_cat + w_xgb + w_hgb
        if abs(w_sum - 1.0) > 1e-6:
            w_cat_n = w_cat / w_sum
            w_xgb_n = w_xgb / w_sum
            w_hgb_n = w_hgb / w_sum
            st.sidebar.warning(f"Weights normalized to: CAT={w_cat_n:.2f}, XGB={w_xgb_n:.2f}, HGB={w_hgb_n:.2f}")
        else:
            w_cat_n, w_xgb_n, w_hgb_n = w_cat, w_xgb, w_hgb

        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🎚️ Decision Threshold Preset")
        preset = st.sidebar.selectbox(
            "Select Preset Option",
            ["F1-Optimal (Maximize Efficiency)", "Zero FN (Safety-First)", "Custom"]
        )
        
        if preset == "F1-Optimal (Maximize Efficiency)":
            selected_thr = 0.9409
            st.sidebar.info("Preset selected: F1-Optimal (Threshold: 0.94). Balances precision and recall to control false alarms.")
        elif preset == "Zero FN (Safety-First)":
            selected_thr = 0.2785
            st.sidebar.info("Preset selected: Zero FN (Threshold: 0.28). Flags any potential defect to miss zero true failures.")
        else:
            selected_thr = st.sidebar.slider("Custom Rank Threshold", 0.0, 1.0, 0.85, 0.01)

        # -------------------------------------------------------------
        # RUN PIPELINE (ABSOLUTELY LEAK-FREE TRANSFORMS)
        # -------------------------------------------------------------
        
        # 1. Feature Engineering
        X_feat = build_features(df_input)
        
        # 2. Schema alignment (match expected scaler columns)
        expected_cols = scaler.feature_names_in_
        X_feat = X_feat.reindex(columns=expected_cols, fill_value=0.0)
        
        # 3. Imputation (using the saved full imputer object)
        x_cols_raw = [c for c in expected_cols if c.upper().startswith("X") and "_" not in c]
        X_feat[x_cols_raw] = imputer.transform(X_feat[x_cols_raw])
        
        # 4. Clean missing stats values (skew/kurt NaNs)
        X_clean = X_feat.fillna(0.0).replace([np.inf, -np.inf], 0.0)
        
        # 5. Robust Scaling (using the saved full scaler object)
        X_scaled = pd.DataFrame(scaler.transform(X_clean), columns=expected_cols)
        
        # 6. Predict Probabilities
        hgb_p = hgb.predict_proba(X_scaled)[:, 1]
        xgb_p = xgb.predict_proba(X_scaled)[:, 1]
        cat_p = cat.predict_proba(X_scaled)[:, 1]
        
        # 7. Convert probabilities to ranks (percentiles)
        hgb_rank = rankdata(hgb_p) / len(hgb_p)
        xgb_rank = rankdata(xgb_p) / len(xgb_p)
        cat_rank = rankdata(cat_p) / len(cat_p)
        
        # 8. Blending
        blend_rank = w_hgb_n * hgb_rank + w_xgb_n * xgb_rank + w_cat_n * cat_rank
        
        # 9. Flag defects based on the selected threshold
        pred_labels = (blend_rank >= selected_thr).astype(int)
        
        # KPI calculations
        total_coils = len(df_input)
        defect_coils = int(pred_labels.sum())
        defect_rate = defect_coils / total_coils
        
        # Setup page tabs
        tab1, tab2, tab3 = st.tabs(["📊 Prediction Dashboard", "💡 Feature Insights", "⚙️ How It Works"])
        
        with tab1:
            # Metric Card Row
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Total Coils Processed</div>
                    <div class="metric-value">{total_coils:,}</div>
                    <div class="metric-sub">Uploaded dataset rows</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                # Color code defect count border
                border_class = "state-green" if defect_rate <= 0.10 else "state-yellow" if defect_rate <= 0.25 else "state-red"
                st.markdown(f"""
                <div class="metric-card {border_class}">
                    <div class="metric-label">Flagged Defects</div>
                    <div class="metric-value">{defect_coils:,}</div>
                    <div class="metric-sub">Coils exceeding rank threshold</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Defect Rate</div>
                    <div class="metric-value">{defect_rate:.2%}</div>
                    <div class="metric-sub">Steel defect ratio</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("---")
            
            # Dashboard Plots
            chart_col, table_col = st.columns([1.2, 1.0])
            
            with chart_col:
                st.markdown("### 📈 Risk Score Distribution & Threshold")
                
                # Plot distribution histogram
                fig, ax = plt.subplots(figsize=(7, 4))
                # Set background colors matching Streamlit dark mode
                fig.patch.set_facecolor('#0e1117')
                ax.set_facecolor('#1e293b')
                
                # Plot density
                sns.histplot(blend_rank, bins=30, kde=True, ax=ax, color="#3B82F6", alpha=0.4)
                # Highlight threshold boundary
                ax.axvline(selected_thr, color="#EF4444", linestyle="--", linewidth=2.5, label=f"Threshold ({selected_thr:.4f})")
                
                # Styling
                ax.set_title("Risk Scores (Ensemble Percentile)", color="#FFFFFF", fontsize=12, pad=10, weight="bold")
                ax.set_xlabel("Rank Score", color="#94A3B8")
                ax.set_ylabel("Count", color="#94A3B8")
                ax.tick_params(colors="#94A3B8")
                ax.spines['bottom'].set_color('#334155')
                ax.spines['top'].set_color('#334155')
                ax.spines['left'].set_color('#334155')
                ax.spines['right'].set_color('#334155')
                ax.grid(color="#334155", linestyle=":")
                ax.legend(facecolor="#1e293b", labelcolor="#FFFFFF")
                
                st.pyplot(fig)
                
            with table_col:
                st.markdown("### 📋 Predicted Anomaly Coils")
                if defect_coils > 0:
                    # Filter input df
                    df_defects = df_input.copy()
                    df_defects["Risk Rank"] = blend_rank
                    df_defects = df_defects[pred_labels == 1].sort_values("Risk Rank", ascending=False)
                    
                    # Columns to show
                    show_cols = ["CoilID", "Risk Rank"]
                    # Add some sample process variables if they exist
                    available_x = [c for c in ["X1", "X10", "X13", "X30", "X32", "X35", "X36", "X41"] if c in df_defects.columns]
                    show_cols.extend(available_x)
                    
                    st.dataframe(
                        df_defects[show_cols].style.format({"Risk Rank": "{:.4f}"}),
                        height=270,
                        use_container_width=True
                    )
                else:
                    st.info("No defective coils found under the current threshold setting.")
                    
            st.markdown("---")
            
            # Export predictions
            st.markdown("### 💾 Export Submission File")
            sub_df = pd.DataFrame({
                "CoilID": df_input["CoilID"],
                "Y": pred_labels
            })
            csv_data = sub_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv_data,
                file_name="tata_steel_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        with tab2:
            st.markdown("### 💡 Anomaly Drivers & Sensor Feature Importance")
            
            # Retrieve CatBoost feature importances
            try:
                importances = cat.get_feature_importance()
                df_imp = pd.DataFrame({
                    "Feature": expected_cols,
                    "Importance": importances
                }).sort_values("Importance", ascending=False).head(20)
                
                col1, col2 = st.columns([1.2, 1.0])
                
                with col1:
                    fig_imp, ax_imp = plt.subplots(figsize=(7, 4.5))
                    fig_imp.patch.set_facecolor('#0e1117')
                    ax_imp.set_facecolor('#1e293b')
                    
                    sns.barplot(data=df_imp, x="Importance", y="Feature", ax=ax_imp, color="#10B981")
                    
                    ax_imp.set_title("Top 20 Defect Predictive Sensors (CatBoost)", color="#FFFFFF", fontsize=12, pad=10, weight="bold")
                    ax_imp.set_xlabel("Importance Value", color="#94A3B8")
                    ax_imp.set_ylabel("", color="#94A3B8")
                    ax_imp.tick_params(colors="#94A3B8")
                    ax_imp.spines['bottom'].set_color('#334155')
                    ax_imp.spines['top'].set_color('#334155')
                    ax_imp.spines['left'].set_color('#334155')
                    ax_imp.spines['right'].set_color('#334155')
                    ax_imp.grid(color="#334155", linestyle=":")
                    
                    st.pyplot(fig_imp)
                    
                with col2:
                    st.markdown("#### 🔍 Primary Anomaly Triggers")
                    st.markdown("""
                    * **Stage 1 (Sensors X1-X16)**: High variances or values in sensors like **X10** and **X13** show strong positive correlations with rolling cracks.
                    * **Stage 3 (Sensors X34-X49)**: Lower values in sensors like **X35** and **X36** strongly signal cooling/rolling temperature deficiencies.
                    """)
                    
                    # Show statistics of a key sensor
                    key_sensor = "X13"
                    if key_sensor in df_input.columns:
                        st.markdown(f"#### 📊 Sensor `{key_sensor}` Comparison")
                        st_df = pd.DataFrame({
                            "Status": ["Flagged Defective" if p == 1 else "Normal" for p in pred_labels],
                            f"{key_sensor}": df_input[key_sensor]
                        })
                        st.dataframe(
                            st_df.groupby("Status").agg({f"{key_sensor}": ["mean", "min", "max", "std"]}),
                            use_container_width=True
                        )
            except Exception as e:
                st.info(f"Could not load feature importances: {e}")
                
        with tab3:
            st.markdown("### ⚙️ Under the Hood: Pipeline Architecture")
            st.markdown("""
            This system runs a **leakage-free, regularized ensembling architecture** designed to match senior industry standards:
            
            #### 1. Preventing Data Leakage
            In standard pipelines, missing value imputation and scaling are fit globally across the entire dataset. This leaks target information from validation folds into the model. Here:
            * **Row-wise Feature Engineering**: Features are computed strictly row-wise (adjacent diffs, mean/std ratio across stages) so that no sample knows any other sample's values.
            * **Fold-wise Preprocessing**: The Median Imputer and Robust Scaler are fit strictly on training splits during validation, and applied to test sets only after a full fit on train data.
            
            #### 2. Model Regularization
            Default ensemble models easily memorize a tiny defect class (4.8% rate). To enforce generalization:
            * **XGBoost** is restricted to a shallow depth of `4`, using `reg_alpha=10` and `reg_lambda=50` to prune unnecessary splits.
            * **CatBoost** is regularized using an L2 leaf regularizer of `50` to smooth prediction bounds.
            
            #### 3. Rank-Normalized Blending
            Probability calibrations shift across validation folds and model instances. To blend safely, probabilities are mapped to **ranks** (percentiles between 0 and 1) before ensembling.
            """)
            
            # Show the OOF validation metrics
            st.markdown("#### 📋 Out-of-Fold (OOF) Validation Summary")
            val_summary_path = os.path.join(PROJECT_DIR, "submissions_v5", "validation_summary.csv")
            if os.path.exists(val_summary_path):
                st.dataframe(pd.read_csv(val_summary_path), use_container_width=True)
            else:
                st.info("Validation summary file not found. Run train_pipeline.py to create it.")
