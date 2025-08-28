# ==========================================
# APORTE PRÁCTICO - MANTENIMIENTO PREDICTIVO
# Dataset: NASA CMAPSS (train_FD001)
# Modelo: LSTM
# ==========================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer

# --------------------------------------------------
# 1. Carga del dataset NASA CMAPSS (FD001)
# --------------------------------------------------
col_names = ["unit_number", "time_in_cycles"] + \
            [f"operational_setting_{i}" for i in range(1, 4)] + \
            [f"sensor_{i}" for i in range(1, 22)]

# ⚠️ Importante: usar train_FD001.txt, no RUL_FD001.txt
train = pd.read_csv("CMAPSSData/train_FD001.txt", sep="\s+", header=None)
train.columns = col_names

# --------------------------------------------------
# 2. Etiquetado del RUL
# --------------------------------------------------
rul = train.groupby("unit_number")["time_in_cycles"].max().reset_index()
rul.columns = ["unit_number", "max_cycle"]
train = train.merge(rul, on="unit_number", how="left")
train["RUL"] = train["max_cycle"] - train["time_in_cycles"]
train.drop("max_cycle", axis=1, inplace=True)

# --------------------------------------------------
# 3. Preprocesamiento
# --------------------------------------------------
sensor_cols = [col for col in train.columns if "sensor" in col]

scaler = MinMaxScaler()
train[sensor_cols] = scaler.fit_transform(train[sensor_cols])

# --------------------------------------------------
# 4. Construcción de ventanas temporales
# --------------------------------------------------
def create_sequences(data, labels, timesteps=30):
    Xs, ys = [], []
    for i in range(len(data)-timesteps):
        Xs.append(data[i:(i+timesteps)])
        ys.append(labels[i+timesteps])
    return np.array(Xs), np.array(ys)

X, y = create_sequences(train[sensor_cols].values, train["RUL"].values, timesteps=30)

X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------------------------------
# 5. Modelo LSTM
# --------------------------------------------------
model = Sequential([
    InputLayer(shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
    LSTM(64, return_sequences=False),
    Dense(32, activation="relu"),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
history = model.fit(X_train_seq, y_train_seq, validation_split=0.2, epochs=10, batch_size=32, verbose=0)

# --------------------------------------------------
# 6. Evaluación
# --------------------------------------------------
y_pred_lstm = model.predict(X_test_seq).flatten()
mae_lstm = mean_absolute_error(y_test_seq, y_pred_lstm)
rmse_lstm = np.sqrt(mean_squared_error(y_test_seq, y_pred_lstm))

results = pd.DataFrame({
    "Modelo": ["Resultado LSTM"],
    "MAE": [mae_lstm],
    "RMSE": [rmse_lstm]
})
print(results)

# --------------------------------------------------
# 7. Gráfica comparativa
# --------------------------------------------------
plt.figure(figsize=(10,5))
plt.plot(y_test_seq[:200], label="Real RUL")
plt.plot(y_pred_lstm[:200], label="Predicción LSTM")
plt.xlabel("Muestras")
plt.ylabel("RUL")
plt.legend()
plt.title("Predicción vs Real - LSTM")
plt.show()
# ==================================================