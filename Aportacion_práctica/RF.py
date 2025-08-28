# ==========================================
# APORTE PRÁCTICO - MANTENIMIENTO PREDICTIVO
# Dataset: NASA CMAPSS (train_FD001)
# Modelo: Random Forest
# ==========================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

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
X = scaler.fit_transform(train[sensor_cols])
y = train["RUL"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------------------------------
# 4. Modelo Random Forest
# --------------------------------------------------
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

# --------------------------------------------------
# 5. Resultados RF
# --------------------------------------------------
results = pd.DataFrame({
    "Modelo": ["Resultado Random Forest"],
    "MAE": [mae_rf],
    "RMSE": [rmse_rf]
})
print(results)

# --------------------------------------------------
# 6. Gráfica comparativa
# --------------------------------------------------
plt.figure(figsize=(10,5))
plt.plot(y_test[:200], label="Real RUL")
plt.plot(y_pred_rf[:200], label="Predicción RF")
plt.xlabel("Muestras")
plt.ylabel("RUL")
plt.legend()
plt.title("Predicción vs Real - Random Forest (subset de test)")
plt.show()
# ==================================================