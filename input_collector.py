import sys
import pandas as pd
import pickle
import numpy as np
from pytorch_tabnet.tab_model import TabNetRegressor
import warnings

warnings.filterwarnings('ignore', category=UserWarning)


INPUT_SCHEMA = [
    ('housing_type', 'Тип жилья', 'categorical', {
        1: 'Новостройка',
        2: 'Вторичка'
    }),
    ('district', 'Район', 'categorical', {
        1: 'Московский',
        2: 'Нижегородский',
        3: 'Канавинский',
        4: 'Сормовский',
        5: 'Автозаводский',
        6: 'Ленинский',
        7: 'Кстовский',
        8: 'Советский',
        9: 'Приокский',
        10: 'Горьковская'
    }),
    ('rooms', 'Количество комнат (1, 2, 3, 4)', 'numeric', None),
    ('is_studio', 'Студия', 'categorical', {
        1: 'Да',
        2: 'Нет'
    }),
    ('total_area', 'Общая площадь (кв.м, например 65.5)', 'numeric', None),
    ('living_area', 'Жилая площадь (кв.м, например 35.2)', 'numeric', None),
    ('kitchen_area', 'Площадь кухни (кв.м, например 10.0)', 'numeric', None),
    ('floor', 'Этаж', 'numeric', None),
    ('num_floors', 'Этажность дома', 'numeric', None),
    ('bathrooms_type', 'Тип санузла', 'categorical', {
        1: 'Раздельный',
        2: 'Совмещенный',
        3: 'Совмещенный, Раздельный'
    }),
    ('num_loggia', 'Количество лоджий', 'numeric', None),
    ('num_balcony', 'Количество балконов', 'numeric', None),
    ('kitchen_and_living', 'Кухня-гостиная', 'categorical', {
        1: 'Да',
        2: 'Нет'
    }),
    ('condition', 'Состояние/Отделка', 'categorical', {
        1: 'Без отделки',
        2: 'Предчистовая',
        3: 'Чистовая',
        4: 'Косметический',
        5: 'Евро',
        6: 'Требует ремонта',
        7: 'Дизайнерский'
    }),
    ('ceiling_height', 'Высота потолков (м, например 2.7)', 'numeric', None),
    ('nearest_metro_st', 'Ближайшая станция метро', 'categorical', {
        1: 'Горьковская',
        2: 'Парк культуры',
        3: 'Чкаловская',
        4: 'Ленинская',
        5: 'Буревестник',
        6: 'Стрелка',
        7: 'Пролетарская',
        8: 'Московская',
        9: 'Заречная',
        10: 'Комсомольская'
    }),
    ('minutes_to_metro', 'Минут до метро', 'numeric', None),
    ('num_freight_lift', 'Количество грузовых лифтов', 'numeric', None),
    ('num_passenger_lift', 'Количество пассажирских лифтов', 'numeric', None),
    ('parking_type', 'Тип парковки', 'categorical', {
        1: 'Открытая',
        2: 'Подземная',
        3: 'Подземная, открытая'
    }),
    ('building_type', 'Тип дома', 'categorical', {
        1: 'Кирпичный',
        2: 'Монолитно-кирпичный',
        3: 'Монолитный',
        4: 'Панельный',
        5: 'Блочный'
    }),
    ('furniture', 'Наличие мебели', 'categorical', {
        1: 'Да',
        2: 'Нет'
    }),
    ('deal_type', 'Тип сделки', 'categorical', {
        1: 'Долевое участие (214-ФЗ)',
        2: 'Свободная продажа',
        3: 'Альтернативная'
    }),
    ('house_completion_year', 'Год сдачи дома', 'numeric', None),
    ('first_floor_is_com', 'Первый этаж коммерческий', 'categorical', {
        1: 'Да',
        2: 'Нет'
    }),
    ('playground', 'Наличие детской площадки', 'categorical', {
        1: 'Да',
        2: 'Нет'
    })
]


def get_user_input():
    print("--- Ввод параметров для модели прогнозирования стоимости жилья ---")
    print("Пожалуйста, вводите данные по очереди.")

    collected_features = {}

    for col_name, prompt, data_type, options in INPUT_SCHEMA:
        while True:
            try:
                if data_type == 'categorical':
                    options_str = ", ".join([f"\n({k}) - {v}" for k, v in options.items()])
                    user_input = input(f"\n{prompt}: {options_str}\nОтвет: ")
                    choice = int(user_input.strip())

                    if choice in options:
                        collected_features[col_name] = options[choice]
                        break
                    else:
                        print("Неверный выбор. Пожалуйста, введите номер из списка.")

                elif data_type == 'numeric':
                    user_input = input(f"\n{prompt}: ")
                    processed_input = user_input.strip().replace(',', '.')

                    value = float(processed_input)
                    if value == int(value):
                        collected_features[col_name] = int(value)
                    else:
                        collected_features[col_name] = value
                    break

            except ValueError:
                print("Неверный формат ввода. Пожалуйста, введите число или выберите номер из списка.")
            except Exception as e:
                print(f"Произошла ошибка: {e}")
                sys.exit(1)

    return collected_features


def main():
    features = get_user_input()
    data = []
    cat_cols = ['district', 'housing_type', 'is_studio', 'bathrooms_type', 'kitchen_and_living', 'condition',
                'nearest_metro_st',
                'parking_type', 'building_type', 'furniture', 'deal_type', 'first_floor_is_com', 'playground']
    print("\n--- Собранные параметры ---")
    for k, v in features.items():
        print(f"{k}: {v}")
    print("\n" + "_"*44)
    if features['num_floors'] > 0:
        features['floor_ratio'] = features['floor'] / features['num_floors']
    else:
        features['floor_ratio'] = 0

    if features['total_area'] > 0:
        features['living_ratio'] = features['living_area'] / features['total_area']
    else:
        features['living_ratio'] = 0

    data = pd.DataFrame([features])
    data_encoded = pd.get_dummies(data, columns=cat_cols, drop_first=True)
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

    print(f"\nПредсказанная цена: {pred_price:,.0f} ₽")


if __name__ == "__main__":
    main()
