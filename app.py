# app.py
# Versi Streamlit dengan penyimpanan model terlatih (joblib)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib
import os

st.set_page_config(layout="wide", page_title="Prediksi Harga BBRI H+1")

st.title("Prediksi Harga Saham BBRI H+1 — Dengan Model Tersimpan (joblib)")

# ------------ Sidebar -------------
uploaded_file = st.sidebar.file_uploader("Upload file CSV (open, high, low, close, volume)", type=["csv"])
file_path_input = st.sidebar.text_input("Atau masukkan path file CSV lokal", value="")
MODEL_DIR = "models"

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# ------------ Load data -------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)
elif file_path_input.strip():
    df = pd.read_csv(file_path_input)
else:
    st.info("Unggah CSV atau masukkan path file lokal.")
    st.stop()

# ------------ Prepare Data -------------
def prepare_dataframe(df_raw):
    df = df_raw.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        df["timestamp"] = pd.RangeIndex(start=0, stop=len(df))

    df["Open_Log"] = np.log(df["open"])
    df["High_Log"] = np.log(df["high"])
    df["Low_Log"] = np.log(df["low"])
    df["Close_Log"] = np.log(df["close"])
    df["Volume_Log"] = np.log(df["volume"] + 1)
    df["MA5_Log"] = df["Close_Log"].rolling(5).mean()

    df["Target_Return"] = df["Close_Log"].shift(-1) - df["Close_Log"]

    df["Feat_Return_1d"] = df["Close_Log"] - df["Close_Log"].shift(1)
    df["Feat_Close_Open"] = df["Close_Log"] - df["Open_Log"]
    df["Feat_High_Low"] = df["High_Log"] - df["Low_Log"]
    df["Feat_Dist_MA5"] = df["Close_Log"] - df["MA5_Log"]
    df["Feat_Vol_Change"] = df["Volume_Log"] - df["Volume_Log"].shift(1)

    df = df.dropna().reset_index(drop=True)
    return df

df = prepare_dataframe(df)
st.dataframe(df.head())

features = [
    "Feat_Return_1d", "Feat_Close_Open", "Feat_High_Low",
    "Feat_Dist_MA5", "Feat_Vol_Change"
]

X = df[features]
y = df["Target_Return"]

test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, shuffle=False
)

test_dates = df.loc[X_test.index, "timestamp"]
test_price_today_log = df.loc[X_test.index, "Close_Log"]

# ------------ Load or Train Models -------------
MODEL_FILES = {
    "scaler": f"{MODEL_DIR}/scaler.pkl",
    "lr": f"{MODEL_DIR}/lr.pkl",
    "rf": f"{MODEL_DIR}/rf.pkl",
    "svr": f"{MODEL_DIR}/svr.pkl",
}

def models_exist():
    return all(os.path.exists(p) for p in MODEL_FILES.values())

if models_exist():
    st.success("📦 Model ditemukan, sedang dimuat...")
    scaler = joblib.load(MODEL_FILES["scaler"])
    models = {
        "Linear Regression": joblib.load(MODEL_FILES["lr"]),
        "Random Forest": joblib.load(MODEL_FILES["rf"]),
        "SVR": joblib.load(MODEL_FILES["svr"]),
    }

else:
    st.warning("⚠️ Model belum ada. Melatih model baru...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {}

    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    models["Linear Regression"] = lr

    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=42)
    rf.fit(X_train, y_train)
    models["Random Forest"] = rf

    svr = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.001)
    svr.fit(X_train_scaled, y_train)
    models["SVR"] = svr

    joblib.dump(scaler, MODEL_FILES["scaler"])
    joblib.dump(lr, MODEL_FILES["lr"])
    joblib.dump(rf, MODEL_FILES["rf"])
    joblib.dump(svr, MODEL_FILES["svr"])

    st.success("✅ Model baru berhasil dilatih & disimpan.")

# ------------ Prediction test set -------------
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

predictions = {
    "Linear Regression": models["Linear Regression"].predict(X_test_scaled),
    "Random Forest": models["Random Forest"].predict(X_test),
    "SVR": models["SVR"].predict(X_test_scaled),
}

actual_log_next = df.loc[X_test.index, "Close_Log"].shift(-1)
actual_price = np.exp(actual_log_next)

results = {}
price_predictions = {}

for name, pred in predictions.items():
    pred_log_next = test_price_today_log + pred
    pred_price = np.exp(pred_log_next)
    price_predictions[name] = pred_price

    mask = ~np.isnan(actual_price)
    y_true = actual_price[mask]
    y_pred = pred_price[mask]

    results[name] = {
        "RMSE (Rp)": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE (Rp)": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }

st.subheader("Evaluasi Model (Rupiah)")
st.dataframe(pd.DataFrame(results))

# ------------ Plot -------------
st.subheader("Plot Prediksi vs Aktual (200 hari terakhir)")
last_n = 200

fig, ax = plt.subplots(figsize=(12, 6))
valid_actual = actual_price[:-1]

ax.plot(test_dates[-last_n:], valid_actual[-last_n:], label="Actual", linewidth=2)

for name, pred in price_predictions.items():
    ax.plot(test_dates[-last_n:], pred[:-1][-last_n:], label=name)

ax.set_title("Prediksi Harga BBRI")
ax.set_xlabel("Tanggal")
ax.set_ylabel("Harga (Rp)")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# ------------ Manual Prediction -------------
st.subheader("Prediksi Manual Harga Besok (H+1)")

col1, col2 = st.columns(2)
with col1:
    open_val = st.number_input("Open", value=float(df["open"].iloc[-1]))
    high_val = st.number_input("High", value=float(df["high"].iloc[-1]))
    low_val = st.number_input("Low", value=float(df["low"].iloc[-1]))
    close_val = st.number_input("Close", value=float(df["close"].iloc[-1]))

with col2:
    vol_val = st.number_input("Volume", value=float(df["volume"].iloc[-1]))
    prev_close = st.number_input("Close Kemarin", value=float(df["close"].iloc[-2]))
    prev_vol = st.number_input("Volume Kemarin", value=float(df["volume"].iloc[-2]))
    ma5 = st.number_input("MA5", value=float(np.exp(df["MA5_Log"].iloc[-1])))

if st.button("Prediksi"):
    log_open = np.log(open_val)
    log_high = np.log(high_val)
    log_low = np.log(low_val)
    log_close = np.log(close_val)
    log_vol = np.log(vol_val + 1)
    log_prev_close = np.log(prev_close)
    log_prev_vol = np.log(prev_vol + 1)
    log_ma5 = np.log(ma5)

    feat = pd.DataFrame([{
        "Feat_Return_1d": log_close - log_prev_close,
        "Feat_Close_Open": log_close - log_open,
        "Feat_High_Low": log_high - log_low,
        "Feat_Dist_MA5": log_close - log_ma5,
        "Feat_Vol_Change": log_vol - log_prev_vol,
    }])

    feat_scaled = scaler.transform(feat)

    pred_lr = models["Linear Regression"].predict(feat_scaled)[0]
    pred_rf = models["Random Forest"].predict(feat)[0]
    pred_svr = models["SVR"].predict(feat_scaled)[0]

    price_lr = close_val * np.exp(pred_lr)
    price_rf = close_val * np.exp(pred_rf)
    price_svr = close_val * np.exp(pred_svr)

    st.write("### Hasil Prediksi:")
    st.write(f"Linear Regression: **Rp {price_lr:,.0f}**")
    st.write(f"Random Forest: **Rp {price_rf:,.0f}**")
    st.write(f"SVR: **Rp {price_svr:,.0f}**")
