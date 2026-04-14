"""
lstm_prototype.py
=================
Light Technologies — AI-Driven Water Disinfection System
Malawi, 2024

This script builds a prototype LSTM model for predicting chlorine residual
from sequences of IoT sensor readings. The core idea is simple: a single
sensor reading doesn't tell you much on its own. What matters is the trend —
how pH, turbidity, and flow rate have been behaving over the past hour or so.
That's exactly what LSTMs are designed to capture.

This is a prototype, not a production model. The architecture is intentionally
kept simple so it's easy to understand, extend, and retrain as more field data
comes in from our NCST pilot deployment.

Target variable : chlorine_residual_mgl (mg/L)
Input sequence  : last N sensor readings (pH, temperature, turbidity, flow rate, resin age)
WHO safe range  : 0.2 – 0.5 mg/L

How to run:
    python lstm_prototype.py                   # generates synthetic data and trains
    python lstm_prototype.py --data your.csv   # use real sensor data

Requirements:
    pip install tensorflow pandas numpy matplotlib scikit-learn
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score

# we import tensorflow lazily inside functions so the script at least
# loads cleanly even if tf isn't installed yet
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ── Config ─────────────────────────────────────────────────────────────────────
# These are the knobs you'll want to tune as real data comes in.
# Keeping them at the top makes that easy.

SEQUENCE_LENGTH = 6       # how many past readings to feed the model (6 x 10min = 1 hour of context)
FEATURES        = ["ph", "temperature_c", "turbidity_ntu", "flow_rate_lpm", "resin_age_days"]
TARGET          = "chlorine_residual_mgl"
WHO_MIN         = 0.2     # mg/L
WHO_MAX         = 0.5     # mg/L
BATCH_SIZE      = 32
EPOCHS          = 100     # early stopping will kick in well before this
RANDOM_SEED     = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ── Synthetic data ─────────────────────────────────────────────────────────────
def make_synthetic_data(n=2000):
    """
    Generates synthetic sensor data that mimics the temporal patterns we see
    in Malawi field deployments — gradual resin degradation, temperature swings
    across the day, turbidity spikes after rainfall events.

    This is a placeholder until the full NCST pilot dataset is ready for release.
    The underlying physics (how each variable affects chlorine release) is based
    on Mtimuni (2024, Stellenbosch University).
    """
    print("[INFO] No data file provided — generating synthetic sensor data.")

    t = np.arange(n)

    # simulate a gradual diurnal temperature cycle (cooler at night, warmer midday)
    temperature = 25 + 8 * np.sin(2 * np.pi * t / 144) + np.random.normal(0, 1.5, n)

    # pH drifts slowly with occasional spikes (e.g. after rainfall changes source water)
    ph = 7.0 + 0.5 * np.sin(2 * np.pi * t / 500) + np.random.normal(0, 0.2, n)
    ph = np.clip(ph, 5.5, 9.0)

    # turbidity spikes randomly — simulating rainfall/runoff events
    turbidity = np.abs(np.random.normal(5, 3, n))
    spike_events = np.random.choice(n, size=15, replace=False)
    for s in spike_events:
        turbidity[s:s+20] += np.random.uniform(15, 40)  # spike lasts ~3 hours
    turbidity = np.clip(turbidity, 0, 50)

    # flow rate varies with usage patterns (higher in morning/evening)
    flow_rate = 1.5 + 0.8 * np.abs(np.sin(2 * np.pi * t / 144)) + np.random.normal(0, 0.2, n)
    flow_rate = np.clip(flow_rate, 0.2, 5.0)

    # resin degrades linearly over its ~180 day lifespan
    resin_age = np.linspace(0, 180, n) + np.random.normal(0, 1, n)
    resin_age = np.clip(resin_age, 0, 200)

    # chlorine residual: physics-informed synthetic target
    # higher pH, turbidity, flow, and resin age all reduce chlorine release
    # temperature slightly accelerates release
    chlorine = (
        0.75
        - 0.04  * (ph - 7.0)
        - 0.003 * turbidity
        - 0.025 * flow_rate
        - 0.002 * resin_age
        + 0.002 * (temperature - 25)
        + np.random.normal(0, 0.025, n)   # sensor noise
    )
    chlorine = np.clip(chlorine, 0.0, 1.2)

    df = pd.DataFrame({
        "timestamp":              pd.date_range("2024-01-01", periods=n, freq="10min"),
        "ph":                     ph.round(2),
        "temperature_c":          temperature.round(1),
        "turbidity_ntu":          turbidity.round(1),
        "flow_rate_lpm":          flow_rate.round(2),
        "resin_age_days":         resin_age.round(0).astype(int),
        "chlorine_residual_mgl":  chlorine.round(3),
    })

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/sample_sensor_data.csv", index=False)
    print(f"[INFO] Synthetic data saved → data/sample_sensor_data.csv  ({n} rows)")
    return df


# ── Data loading ───────────────────────────────────────────────────────────────
def load_data(path):
    df = pd.read_csv(path)

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"[ERROR] These columns are missing from your CSV: {missing}")

    before = len(df)
    df = df.dropna(subset=FEATURES + [TARGET])
    dropped = before - len(df)
    if dropped > 0:
        print(f"[WARN] Dropped {dropped} rows with missing values.")

    print(f"[INFO] Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
    return df


# ── Sequence builder ───────────────────────────────────────────────────────────
def build_sequences(feature_arr, target_arr, seq_len):
    """
    Converts flat arrays into overlapping input/output sequences.

    For each position i, we take the previous seq_len readings as input
    and ask the model to predict the chlorine residual at position i.

    Example with seq_len=6:
        Input:  readings at t-6, t-5, t-4, t-3, t-2, t-1
        Output: chlorine residual at t
    """
    X, y = [], []
    for i in range(seq_len, len(feature_arr)):
        X.append(feature_arr[i - seq_len:i])
        y.append(target_arr[i])
    return np.array(X), np.array(y)


# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess(df):
    """
    Scale features to [0, 1] — LSTMs are sensitive to input scale.
    We keep separate scalers for features and target so we can
    inverse-transform predictions back to mg/L at the end.
    """
    feature_scaler = MinMaxScaler()
    target_scaler  = MinMaxScaler()

    features_scaled = feature_scaler.fit_transform(df[FEATURES].values)
    target_scaled   = target_scaler.fit_transform(df[[TARGET]].values).flatten()

    X, y = build_sequences(features_scaled, target_scaled, SEQUENCE_LENGTH)

    # 80/20 train/test split — keep temporal order (no shuffling)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"[INFO] Training sequences : {len(X_train)}")
    print(f"[INFO] Test sequences     : {len(X_test)}")
    print(f"[INFO] Input shape        : {X_train.shape}  (samples, timesteps, features)")

    return X_train, X_test, y_train, y_test, target_scaler


# ── Model definition ───────────────────────────────────────────────────────────
def build_model(seq_len, n_features):
    """
    Two-layer LSTM with dropout for regularisation.

    The architecture is deliberately modest — we don't have tens of thousands
    of training sequences yet. As the NCST pilot generates more data, we can
    add layers, increase units, or move to a more complex architecture like
    a bidirectional LSTM or Transformer.

    For now, this is the right size for the data we have.
    """
    model = Sequential([
        # first LSTM layer — return_sequences=True passes the full sequence
        # to the next LSTM layer rather than just the final hidden state
        LSTM(64, return_sequences=True, input_shape=(seq_len, n_features)),
        Dropout(0.2),   # randomly zero out 20% of units to prevent overfitting

        # second LSTM layer — summarises the sequence into a single vector
        LSTM(32, return_sequences=False),
        Dropout(0.2),

        # fully connected output layer — single chlorine residual value
        Dense(16, activation="relu"),
        Dense(1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",          # mean squared error — standard for regression
        metrics=["mae"]
    )

    return model


# ── Training ───────────────────────────────────────────────────────────────────
def train(model, X_train, y_train):
    """
    We use two callbacks:
    - EarlyStopping: stops training if validation loss stops improving.
      This is important — we don't want to overfit on a small dataset.
    - ReduceLROnPlateau: reduces the learning rate when training stalls.
      Helps squeeze out the last bit of performance without manual tuning.
    """
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=12,          # stop after 12 epochs with no improvement
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,           # halve the learning rate when stuck
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    print("\n[INFO] Training LSTM model...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.15,    # hold out 15% of training data for validation
        callbacks=callbacks,
        verbose=1
    )

    print(f"[INFO] Training stopped at epoch {len(history.history['loss'])}")
    return history


# ── Evaluation ─────────────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test, target_scaler):
    """
    Evaluate in original mg/L units (after inverse-transforming the scaled predictions).
    We also check WHO compliance — this is the metric that actually matters for
    whether the system is keeping water safe.
    """
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()

    # inverse transform back to mg/L
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    pred_safe   = (y_pred >= WHO_MIN) & (y_pred <= WHO_MAX)
    actual_safe = (y_true >= WHO_MIN) & (y_true <= WHO_MAX)
    who_acc     = (pred_safe == actual_safe).mean() * 100

    print("\n" + "═" * 50)
    print("  LSTM MODEL — TEST RESULTS")
    print("═" * 50)
    print(f"  MAE              : {mae:.4f} mg/L")
    print(f"  RMSE             : {rmse:.4f} mg/L")
    print(f"  R² Score         : {r2:.4f}")
    print(f"  WHO Compliance   : {who_acc:.1f}% classification accuracy")
    print("═" * 50)

    # flag if we're not hitting the target R² yet
    if r2 < 0.70:
        print("  [WARN] R² below 0.70 — model needs more data or architecture tuning.")
    elif r2 >= 0.90:
        print("  [GOOD] R² ≥ 0.90 — hitting project target accuracy.")
    else:
        print("  [OK]   R² between 0.70–0.90 — reasonable for current dataset size.")

    return y_true, y_pred, {"mae": mae, "rmse": rmse, "r2": r2, "who_acc": who_acc}


# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_results(history, y_true, y_pred, metrics, save_dir="outputs/"):
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Light Technologies — LSTM Chlorine Residual Prediction", fontsize=13)

    # Plot 1: Training loss curve
    axes[0].plot(history.history["loss"],     label="Train loss",      color="steelblue")
    axes[0].plot(history.history["val_loss"], label="Validation loss", color="darkorange", linestyle="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Training History")
    axes[0].legend()

    # Plot 2: Actual vs predicted over time (first 300 test points)
    n_show = min(300, len(y_true))
    axes[1].plot(y_true[:n_show],  label="Actual",    color="steelblue",  linewidth=1.2)
    axes[1].plot(y_pred[:n_show],  label="Predicted", color="darkorange", linewidth=1.2, linestyle="--")
    axes[1].axhspan(WHO_MIN, WHO_MAX, alpha=0.12, color="green", label="WHO safe range")
    axes[1].set_xlabel("Time step (×10 min)")
    axes[1].set_ylabel("Chlorine Residual (mg/L)")
    axes[1].set_title("Predicted vs Actual — Test Set")
    axes[1].legend(fontsize=8)

    # Plot 3: Scatter — actual vs predicted
    axes[2].scatter(y_true, y_pred, alpha=0.3, s=15, color="teal")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[2].plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    axes[2].axhspan(WHO_MIN, WHO_MAX, alpha=0.1, color="green", label="WHO safe range")
    axes[2].set_xlabel("Actual (mg/L)")
    axes[2].set_ylabel("Predicted (mg/L)")
    axes[2].set_title(f"Scatter Plot\nR² = {metrics['r2']:.3f}  |  MAE = {metrics['mae']:.3f} mg/L")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(save_dir, "lstm_results.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n[INFO] Plots saved → {out}")
    plt.show()


# ── Save model ─────────────────────────────────────────────────────────────────
def save_model(model, save_dir="outputs/"):
    """
    Save in two formats:
    - Keras native (.keras) for continued training and experimentation
    - TFLite (.tflite) for deployment on edge IoT devices in the field
      (low memory, no internet required — critical for rural Malawi deployments)
    """
    os.makedirs(save_dir, exist_ok=True)

    keras_path = os.path.join(save_dir, "lstm_chlorine_model.keras")
    model.save(keras_path)
    print(f"[INFO] Keras model saved  → {keras_path}")

    # TFLite conversion for edge deployment
    try:
        converter   = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        tflite_path  = os.path.join(save_dir, "lstm_chlorine_model.tflite")
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        tflite_kb = os.path.getsize(tflite_path) / 1024
        print(f"[INFO] TFLite model saved → {tflite_path}  ({tflite_kb:.1f} KB)")
        print("       This is the version that runs on field IoT devices.")
    except Exception as e:
        print(f"[WARN] TFLite conversion failed: {e}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Light Technologies — LSTM Chlorine Residual Model")
    parser.add_argument("--data",   type=str, default=None,       help="Path to sensor CSV")
    parser.add_argument("--outdir", type=str, default="outputs/", help="Output directory")
    args = parser.parse_args()

    print("\n" + "═" * 50)
    print("  Light Technologies — LSTM Prototype")
    print("  Chlorine Residual Time-Series Prediction")
    print("  Malawi, 2024")
    print("═" * 50)

    # load or generate data
    if args.data and os.path.exists(args.data):
        df = load_data(args.data)
    else:
        df = make_synthetic_data(n=2000)
        df = load_data("data/sample_sensor_data.csv")

    # preprocess → sequences
    X_train, X_test, y_train, y_test, target_scaler = preprocess(df)

    # build model
    model = build_model(seq_len=SEQUENCE_LENGTH, n_features=len(FEATURES))
    model.summary()

    # train
    history = train(model, X_train, y_train)

    # evaluate
    y_true, y_pred, metrics = evaluate(model, X_test, y_test, target_scaler)

    # plots
    plot_results(history, y_true, y_pred, metrics, save_dir=args.outdir)

    # save
    save_model(model, save_dir=args.outdir)

    print("\n[DONE] LSTM prototype complete.")
    print("       Next: integrate real NCST pilot data and tune SEQUENCE_LENGTH.")
    print("       The .tflite model in outputs/ is ready for edge device testing.")


if __name__ == "__main__":
    main()
