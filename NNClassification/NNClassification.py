import numpy as np
import tensorflow as tf
import os
import pickle
from sklearn.preprocessing import StandardScaler

# Определяем путь к текущему файлу (NNClassification.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Пути к файлам относительно директории NNClassification
MODEL_PATH = os.path.join(BASE_DIR, "risk_classification_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

def load_model():
    """Загружает обученную модель и scaler."""
    print(f"Current working directory: {os.getcwd()}")  # Отладка
    print(f"Looking for model at: {MODEL_PATH}")  # Отладка
    print(f"Looking for scaler at: {SCALER_PATH}")  # Отладка
    
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        if os.path.exists(SCALER_PATH):
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
        else:
            raise FileNotFoundError(f"Scaler не найден по пути: {SCALER_PATH}")
    else:
        raise FileNotFoundError(f"Модель не найдена по пути: {MODEL_PATH}")
    return model, scaler

def predict_risk(stocks, bonds, crypto, metals, etf, horizon):
    """Предсказывает риск портфеля на основе долей активов и горизонта."""
    # Проверка суммы долей активов
    total_weight = stocks + bonds + crypto + metals + etf
    if not abs(total_weight - 1.0) < 1e-5:
        raise ValueError("Сумма долей активов должна быть равна 1.0")

    # Проверка горизонта
    if not isinstance(horizon, int) or horizon < 1 or horizon > 10:
        raise ValueError("Горизонт инвестирования должен быть целым числом от 1 до 10.")

    # Расчет волатильности
    volatility = (
        stocks * np.random.uniform(0.20, 0.30) +
        crypto * np.random.uniform(0.30, 0.60) +
        bonds * np.random.uniform(0.01, 0.05) +
        metals * np.random.uniform(0.10, 0.20) +
        etf * np.random.uniform(0.15, 0.25)
    )

    # Загрузка модели и scaler
    model, scaler = load_model()

    # Подготовка данных для предсказания
    input_data = np.array([[stocks, bonds, crypto, metals, etf, volatility, horizon]])
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data, verbose=0)
    # Температурное масштабирование
    temperature = 2.0
    scaled_logits = prediction / temperature
    probabilities = tf.nn.softmax(scaled_logits).numpy()[0]
    risk_class = np.argmax(probabilities, axis=0)

    risk_labels = ["Низкий", "Умеренный", "Высокий"]
    return risk_labels[risk_class], probabilities

if __name__ == "__main__":
    # Пример использования
    risk, probs = predict_risk(0.4, 0.3, 0.1, 0.1, 0.1, 4)
    print(f"Итоговый риск: {risk}")
    print(f"Вероятности: {probs}")