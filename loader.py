import pickle

import pandas as pd
import numpy as np
from pytorch_tabnet.tab_model import TabNetRegressor

test_data = {
    "housing_type": "Новостройка",
    "district": "Канавинский",
    "rooms": 3,
    "is_studio": "Нет",
    "total_area": 62.2,
    "living_area": 29.2,
    "kitchen_area": 9.9,
    "floor": 10,
    "num_floors": 25,
    "bathrooms_type": "Раздельный",
    "num_loggia": 1,
    "num_balcony": 0,
    "kitchen_and_living": "Да",
    "condition": "Предчистовая",
    "ceiling_height": 3.0,
    "nearest_metro_st": "Чкаловская",
    "minutes_to_metro": 1,
    "num_freight_lift": 1,
    "num_passenger_lift": 1,
    "parking_type": "Подземная",
    "building_type": "Монолитно-кирпичный",
    "furniture": "Нет",
    "deal_type": "Долевое участие (214-ФЗ)",
    "house_completion_year": 2027,
    "first_floor_is_com": "Да",
    "playground": "Да",
    "floor_ratio": 10 / 25,
    "living_ratio": 29.2 / 62.2
}

user_df = pd.DataFrame([test_data])
cat_cols = ['district', 'housing_type', 'is_studio', 'bathrooms_type', 'kitchen_and_living', 'condition', 'nearest_metro_st',
            'parking_type', 'building_type', 'furniture', 'deal_type', 'first_floor_is_com', 'playground']
data_encoded = pd.get_dummies(user_df, columns=cat_cols, drop_first=True)
print(data_encoded.head(5))
for col in data_encoded.columns:
    if data_encoded[col].dtype in ['object', 'bool']:
        data_encoded[col] = data_encoded[col].astype(float)

data_encoded.replace([np.inf, -np.inf], np.nan, inplace=True)
data_encoded.fillna(0, inplace=True)
feature_cols = [c for c in data_encoded.columns if c != "price"]

with open("feature_order.pkl", "rb") as f:
    feature_order = pickle.load(f)

for col in feature_order:
    if col not in data_encoded.columns:
        data_encoded[col] = 0

data_encoded = data_encoded[feature_order]


with open("scaler_X.pkl", "rb") as f:
    scaler_X = pickle.load(f)

with open("scaler_y.pkl", "rb") as f:
    scaler_y = pickle.load(f)

tabnet = TabNetRegressor()
tabnet.load_model("tabnet_model.zip")

X_scaled = scaler_X.transform(data_encoded)

pred_scaled = tabnet.predict(X_scaled).flatten()
pred_price = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]

print(f"Предсказанная цена: {pred_price:,.0f} ₽")