import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from pytorch_tabnet.tab_model import TabNetRegressor

def fill_missing(data):
    for col in data.columns:
        if data[col].dtype in ['float64', 'int64', 'float32', 'int32']:
            data[col].fillna(data[col].mean(), inplace=True)
        else:
            data[col].fillna(data[col].mode()[0], inplace=True)
    return data

def metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MSE": mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }


# ЭТАП 1 -- ОБРАБОТКА ДАННЫХ

data = pd.read_csv("Dataset_5000.csv", sep=";")
data = fill_missing(data)
data = data.apply(lambda col: col.map(lambda x: str(x).replace("\xa0", "").strip() if pd.notna(x) else x))
cols_with_commas = [
    "rooms", "total_area", "living_area", "kitchen_area", "num_floors", "floor",
    "num_loggia", "num_balcony", "num_freight_lift", "num_passenger_lift", "house_completion_year", "ceiling_height"]

for col in cols_with_commas:
    if col in data.columns:
        data[col] = data[col].map(lambda x: str(x).replace(",", ".").replace(" ", "") if pd.notna(x) else x)
        data[col] = pd.to_numeric(data[col], errors="coerce")
data['price'] = data['price'].str.replace(" ", "")
#print(data['price'])
data["price"] = pd.to_numeric(data["price"], errors="coerce")
#print(data['price'])
data.dropna(subset=["price"], inplace=True)
# FEATURE ENGINEERING !!!
data["floor_ratio"] = data["floor"] / data["num_floors"].replace(0, np.nan)
data["living_ratio"] = data["living_area"] / data["total_area"].replace(0, np.nan)

#data.fillna(0, inplace=True)
print("Типы данных в исходном DataFrame:")
print(data.dtypes)
cat_cols = data.select_dtypes(include=["object"]).columns.tolist()
data_encoded = pd.get_dummies(data, columns=cat_cols, drop_first=True)

for col in data_encoded.columns:
    if data_encoded[col].dtype in ['object', 'bool']:
        data_encoded[col] = data_encoded[col].astype(float)

data_encoded.replace([np.inf, -np.inf], np.nan, inplace=True)
data_encoded.fillna(0, inplace=True)
feature_cols = [c for c in data_encoded.columns if c != "price"]

# ЭТАП 2 -- Выборки
X = data_encoded[feature_cols].astype(np.float32)
y = data_encoded["price"].astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

feature_order = X_train.columns.tolist()

with open("feature_order.pkl", "wb") as f:
    pickle.dump(feature_order, f)

with open("scaler_X.pkl", "wb") as f:
    pickle.dump(scaler_X, f)

with open("scaler_y.pkl", "wb") as f:
    pickle.dump(scaler_y, f)

# ЭТАП 3 -- ОБУЧЕНИЕ МОДЕЛЕЙ

# MLP базовая

model_mlp_base = Sequential([
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])
model_mlp_base.compile(optimizer='adam', loss='mse')

model_mlp_base.fit(
    X_train_scaled, y_train_scaled,
    epochs=50, batch_size=32,
    validation_split=0.2,
    verbose=1
)

# MLP улучшенная

model_mlp = Sequential([
    Dense(256),
    LeakyReLU(),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128),
    LeakyReLU(),
    BatchNormalization(),
    Dropout(0.2),
    Dense(64),
    LeakyReLU(),
    Dropout(0.2),
    Dense(32),
    LeakyReLU(),
    Dense(1)
])

model_mlp.compile(optimizer='adam', loss='mse')
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)

model_mlp.fit(
    X_train_scaled, y_train_scaled,
    epochs=300, batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop, lr_reduce],
    verbose=1
)

# TABNET

tabnet = TabNetRegressor(
    n_d=128, n_a=128, n_steps=7,
    gamma=1.5, lambda_sparse=1e-3,
    n_shared=2, n_independent=2,
    optimizer_params=dict(lr=1e-2),
    mask_type='entmax'
)

tabnet.fit(
    X_train_scaled, y_train_scaled.reshape(-1, 1),
    eval_set=[(X_test_scaled, y_test_scaled.reshape(-1, 1))],
    eval_metric=["rmse"],
    patience=100,
    max_epochs=1000,
    batch_size=256,
    virtual_batch_size=64,
    num_workers=0,
    drop_last=False
)

# ЭТАП 4 -- СОХРАНЕНИЕ МОДЕЛЕЙ

model_mlp_base.save("mlp_base.keras")
model_mlp.save("mlp_adv.keras")
tabnet.save_model("tabnet_model")

pred_base_scaled = model_mlp_base.predict(X_test_scaled).flatten()
pred_base = scaler_y.inverse_transform(pred_base_scaled.reshape(-1, 1)).flatten()

pred_mlp_scaled = model_mlp.predict(X_test_scaled).flatten()
pred_mlp = scaler_y.inverse_transform(pred_mlp_scaled.reshape(-1, 1)).flatten()

pred_tabnet_scaled = tabnet.predict(X_test_scaled).flatten()
pred_tabnet = scaler_y.inverse_transform(pred_tabnet_scaled.reshape(-1, 1)).flatten()

# ЭТАП 5 -- ОЦЕНКА КАЧЕСТВА

results = pd.DataFrame({
    "MLP Базовая": metrics(y_test, pred_base),
    "MLP Улучшенная": metrics(y_test, pred_mlp),
    "TabNet": metrics(y_test, pred_tabnet)
}).T

print("Сравнение моделей:\n")
print(results)

results.to_csv("results.csv", index=False)
results.to_excel("results.xlsx", index=False)
results.to_json("results.json", orient="records", indent=2)

