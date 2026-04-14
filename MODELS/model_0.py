"""
baseline_regression.py
======================
Light Technologies — AI-Driven Water Disinfection System
Baseline ML model for predicting chlorine residual from IoT sensor inputs.

This script trains and evaluates two baseline models:
  1. Linear Regression       — interpretable benchmark
  2. Random Forest Regressor — stronger non-linear baseline

Target variable : chlorine_residual_mgl  (mg/L)
Features        : pH, temperature_c, turbidity_ntu, flow_rate_lpm, resin_age_days

WHO safe range  : 0.2 – 0.5 mg/L

Usage:
    python baseline_regression.py                 
    python baseline_regression.py --data path/to/sensor_data.csv

Requirements:
    pip install pandas scikit-learn matplotlib seaborn
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

# ── Constants ──────────────────────────────────────────────────────────────────
WHO_MIN = 0.2   # mg/L minimum safe chlorine residual
WHO_MAX = 0.5   # mg/L maximum safe chlorine residual
RANDOM_STATE = 42
FEATURE_COLS = ["ph", "temperature_c", "turbidity_ntu", "flow_rate_lpm", "resin_age_days"]
TARGET_COL   = "chlorine_residual_mgl"


# ── 1. Synthetic Data Generator ────────────────────────────────────────────────
def generate_synthetic_data(n_samples: int = 1000, save_path: str = "data/sample_sensor_data.csv") -> pd.DataFrame:
    """
    Generates realistic synthetic sensor data when no real CSV is provided.
    Mimics real-world variability observed in Malawi field deployments.
    Replace with real NCST pilot data when available.
    """
    np.random.seed(RANDOM_STATE)

    ph              = np.random.uniform(5.5, 9.0,  n_samples)       # pH range in Malawi water sources
    temperature_c   = np.random.uniform(10.0, 40.0, n_samples)      # seasonal temperature range (°C)
    turbidity_ntu   = np.random.uniform(0.0, 50.0,  n_samples)      # NTU — post-flood values can be high
    flow_rate_lpm   = np.random.uniform(0.5, 5.0,   n_samples)      # litres per minute
    resin_age_days  = np.random.uniform(0.0, 180.0, n_samples)      # resin cartridge age in days

    # Physics-informed synthetic target:
    # Chlorine release decreases with higher pH, turbidity, flow rate, and resin age
    # Increases slightly with temperature (accelerated release)
    chlorine_residual = (
        0.80
        - 0.05  * (ph - 7.0)
        - 0.004 * turbidity_ntu
        - 0.03  * flow_rate_lpm
        - 0.002 * resin_age_days
        + 0.003 * (temperature_c - 25.0)
        + np.random.normal(0, 0.04, n_samples)   # sensor noise
    )

    # Clip to physically plausible range
    chlorine_residual = np.clip(chlorine_residual, 0.0, 1.2)

    df = pd.DataFrame({
        "timestamp":             pd.date_range("2024-01-01", periods=n_samples, freq="10min"),
        "ph":                    ph.round(2),
        "temperature_c":         temperature_c.round(1),
        "turbidity_ntu":         turbidity_ntu.round(1),
        "flow_rate_lpm":         flow_rate_lpm.round(2),
        "resin_age_days":        resin_age_days.round(0).astype(int),
        "chlorine_residual_mgl": chlorine_residual.round(3),
    })

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"[INFO] Synthetic dataset saved to: {save_path}  ({n_samples} rows)")
    return df


# ── 2. Data Loading & Validation ───────────────────────────────────────────────
def load_and_validate(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"[ERROR] Missing columns in dataset: {missing}")

    before = len(df)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    print(f"[INFO] Loaded {before} rows — {before - len(df)} dropped due to missing values.")

    # Flag WHO compliance for analysis
    df["who_compliant"] = df[TARGET_COL].between(WHO_MIN, WHO_MAX)
    compliance_pct = df["who_compliant"].mean() * 100
    print(f"[INFO] WHO compliance in dataset: {compliance_pct:.1f}% of readings")
    return df


# ── 3. Model Training & Evaluation ────────────────────────────────────────────
def evaluate_model(name: str, model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    y_pred = model.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    r2     = r2_score(y_test, y_pred)

    # WHO compliance accuracy: did model correctly predict safe vs. unsafe?
    pred_compliant   = pd.Series(y_pred).between(WHO_MIN, WHO_MAX)
    actual_compliant = pd.Series(y_test).between(WHO_MIN, WHO_MAX)
    who_accuracy     = (pred_compliant == actual_compliant.values).mean() * 100

    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(f"  MAE              : {mae:.4f} mg/L")
    print(f"  RMSE             : {rmse:.4f} mg/L")
    print(f"  R² Score         : {r2:.4f}")
    print(f"  WHO Compliance   : {who_accuracy:.1f}% classification accuracy")

    return {"name": name, "mae": mae, "rmse": rmse, "r2": r2,
            "who_accuracy": who_accuracy, "y_pred": y_pred}


def train_models(df: pd.DataFrame) -> dict:
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # ── Model 1: Linear Regression (with scaling) ──────────────────────────
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LinearRegression())
    ])
    lr_pipeline.fit(X_train, y_train)

    cv_lr = cross_val_score(lr_pipeline, X_train, y_train, cv=5, scoring="r2")
    print(f"\n[Linear Regression] 5-Fold CV R²: {cv_lr.mean():.4f} ± {cv_lr.std():.4f}")
    lr_results = evaluate_model("Linear Regression", lr_pipeline, X_test, y_test)

    # ── Model 2: Random Forest ─────────────────────────────────────────────
    rf_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])
    rf_pipeline.fit(X_train, y_train)

    cv_rf = cross_val_score(rf_pipeline, X_train, y_train, cv=5, scoring="r2")
    print(f"\n[Random Forest] 5-Fold CV R²: {cv_rf.mean():.4f} ± {cv_rf.std():.4f}")
    rf_results = evaluate_model("Random Forest", rf_pipeline, X_test, y_test)

    # ── Feature Importance (Random Forest) ────────────────────────────────
    perm_imp = permutation_importance(
        rf_pipeline, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE
    )
    importance_df = pd.DataFrame({
        "feature":    FEATURE_COLS,
        "importance": perm_imp.importances_mean
    }).sort_values("importance", ascending=False)

    print("\n[Random Forest] Permutation Feature Importance:")
    print(importance_df.to_string(index=False))

    return {
        "lr":  {"pipeline": lr_pipeline, "results": lr_results},
        "rf":  {"pipeline": rf_pipeline, "results": rf_results, "importance": importance_df},
        "X_test": X_test, "y_test": y_test
    }


# ── 4. Visualisations ──────────────────────────────────────────────────────────
def plot_results(output: dict, save_dir: str = "outputs/"):
    os.makedirs(save_dir, exist_ok=True)

    y_test    = output["y_test"]
    lr_pred   = output["lr"]["results"]["y_pred"]
    rf_pred   = output["rf"]["results"]["y_pred"]
    imp_df    = output["rf"]["importance"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Light Technologies — Chlorine Residual Prediction (Baseline Models)", fontsize=13)

    # Plot 1: Actual vs. Predicted — Linear Regression
    axes[0].scatter(y_test, lr_pred, alpha=0.4, color="steelblue", s=20)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=1.5)
    axes[0].axhspan(WHO_MIN, WHO_MAX, alpha=0.1, color="green", label="WHO safe range")
    axes[0].set_xlabel("Actual Chlorine (mg/L)")
    axes[0].set_ylabel("Predicted Chlorine (mg/L)")
    axes[0].set_title(f"Linear Regression\nR² = {output['lr']['results']['r2']:.3f}")
    axes[0].legend(fontsize=8)

    # Plot 2: Actual vs. Predicted — Random Forest
    axes[1].scatter(y_test, rf_pred, alpha=0.4, color="darkorange", s=20)
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=1.5)
    axes[1].axhspan(WHO_MIN, WHO_MAX, alpha=0.1, color="green", label="WHO safe range")
    axes[1].set_xlabel("Actual Chlorine (mg/L)")
    axes[1].set_ylabel("Predicted Chlorine (mg/L)")
    axes[1].set_title(f"Random Forest\nR² = {output['rf']['results']['r2']:.3f}")
    axes[1].legend(fontsize=8)

    # Plot 3: Feature Importance
    axes[2].barh(imp_df["feature"], imp_df["importance"], color="teal")
    axes[2].set_xlabel("Permutation Importance")
    axes[2].set_title("Feature Importance\n(Random Forest)")
    axes[2].invert_yaxis()

    plt.tight_layout()
    out_path = os.path.join(save_dir, "baseline_model_results.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n[INFO] Plot saved to: {out_path}")
    plt.show()


# ── 5. WHO Compliance Report ───────────────────────────────────────────────────
def who_compliance_report(df: pd.DataFrame):
    total      = len(df)
    compliant  = df["who_compliant"].sum()
    under      = (df[TARGET_COL] < WHO_MIN).sum()
    over       = (df[TARGET_COL] > WHO_MAX).sum()

    print(f"\n{'═'*50}")
    print(f"  WHO COMPLIANCE REPORT")
    print(f"{'═'*50}")
    print(f"  Total readings   : {total}")
    print(f"  Compliant        : {compliant} ({compliant/total*100:.1f}%)")
    print(f"  Under-dosed (<{WHO_MIN}): {under} ({under/total*100:.1f}%)  ← UNSAFE")
    print(f"  Over-dosed  (>{WHO_MAX}): {over}  ({over/total*100:.1f}%)  ← UNSAFE")
    print(f"{'═'*50}")


# ── 6. Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Light Technologies — Chlorine Residual Baseline Model")
    parser.add_argument("--data",    type=str, default=None,    help="Path to sensor CSV file")
    parser.add_argument("--samples", type=int, default=1000,    help="Synthetic samples if no data file")
    parser.add_argument("--outdir",  type=str, default="outputs/", help="Directory for output plots")
    args = parser.parse_args()

    print("\n" + "═"*50)
    print("  Light Technologies — Baseline Chlorine Model")
    print("  AI-Driven Water Disinfection | Malawi")
    print("═"*50)

    # Load or generate data
    if args.data and os.path.exists(args.data):
        print(f"[INFO] Loading data from: {args.data}")
        df = load_and_validate(args.data)
    else:
        print("[INFO] No data file provided — generating synthetic data.")
        df = generate_synthetic_data(n_samples=args.samples)
        df = load_and_validate("data/sample_sensor_data.csv")

    who_compliance_report(df)

    # Train and evaluate
    output = train_models(df)

    # Visualise
    plot_results(output, save_dir=args.outdir)

    print("\n[DONE] Baseline models complete.")
    print("       Next step: lstm_prototype.py for temporal sequence modelling.")


if __name__ == "__main__":
    main()
