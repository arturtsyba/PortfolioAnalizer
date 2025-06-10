import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta

# --- Загрузка данных ---
def load_portfolio_data(tickers, years=5, interval='1d'):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365)
    data = []
    for ticker in tickers:
        df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
        if df.empty:
            raise ValueError(f"Нет данных для {ticker}")
        df = df[['Close']].rename(columns={'Close': ticker})
        data.append(df)
    df = pd.concat(data, axis=1, join='inner').dropna()
    if df.empty:
        raise ValueError("Нет общих данных для тикеров после объединения")
    print(f"Загружено {len(df)} строк для портфеля с {start_date.date()} по {end_date.date()}")
    return df

# --- Технические индикаторы ---
def calculate_rsi(data, periods=14):
    delta = data.diff().fillna(0)
    gain = delta.where(delta > 0, 0).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def calculate_ema(data, periods=10):
    return data.ewm(span=periods, adjust=False).mean()

# --- Подготовка признаков ---
def prepare_features(df, tickers, weights):
    df_features = pd.DataFrame(index=df.index)
    
    # Доходности и волатильность активов
    returns = df.pct_change().dropna()
    for ticker in tickers:
        return_series = returns[ticker].squeeze()
        df_features[f'{ticker}_return'] = return_series
        df_features[f'{ticker}_volatility'] = return_series.rolling(20).std() * np.sqrt(252)
        df_features[f'{ticker}_rsi'] = calculate_rsi(df[ticker])
        df_features[f'{ticker}_ema'] = calculate_ema(df[ticker]) / df[ticker]
    
    # Корреляции между активами
    for i, ticker1 in enumerate(tickers):
        for ticker2 in tickers[i+1:]:
            series1 = returns[ticker1].squeeze()
            series2 = returns[ticker2].squeeze()
            corr_series = pd.Series(series1.rolling(20).corr(series2), index=returns.index)
            df_features[f'corr_{ticker1}_{ticker2}'] = corr_series
    
    # Макроэкономические данные
    try:
        vix = yf.download('^VIX', start=df.index[0], end=df.index[-1], interval='1d', progress=False)['Close']
        vix = vix.reindex(df.index, method='ffill')
        df_features['VIX'] = vix / 100.0
    except:
        df_features['VIX'] = 0.2  # Запасное значение
    
    # Нормализация признаков
    scaler_features = StandardScaler()
    feature_cols = [col for col in df_features.columns if col != 'portfolio_return']
    df_features[feature_cols] = scaler_features.fit_transform(df_features[feature_cols])
    
    # Целевая переменная: дневная доходность портфеля
    portfolio_returns = (returns * weights).sum(axis=1)
    df_features['portfolio_return'] = portfolio_returns.shift(-1)
    
    df_features = df_features.dropna()
    print("Размер данных после очистки:", len(df_features))
    return df_features, scaler_features, feature_cols

# --- Подготовка данных для LSTM ---
def create_sequences(X, y, timesteps=40):
    X_seq, y_seq = [], []
    for i in range(len(X) - timesteps):
        X_seq.append(X[i:i+timesteps])
        y_seq.append(y[i+timesteps])
    return np.array(X_seq), np.array(y_seq)

# --- Модель с LSTM ---
def build_lstm_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True,
                                                           kernel_regularizer=tf.keras.regularizers.l2(0.04))),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=False,
                                                           kernel_regularizer=tf.keras.regularizers.l2(0.04))),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Dense(32),
        tf.keras.layers.LeakyReLU(negative_slope=0.1),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mae', metrics=['mae'])
    return model

# --- Callback для прогресс-бара ---
class ProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback
        self.total_epochs = 0

    def on_train_begin(self, logs=None):
        self.total_epochs = self.params['epochs']

    def on_epoch_end(self, epoch, logs=None):
        if self.callback:
            self.callback(epoch + 1, self.total_epochs)

# --- Анализ портфеля ---
def analyze_portfolio(tickers, weights, portfolio_name, timesteps=40, callback=None):
    try:
        print(f"\nАнализ портфеля '{portfolio_name}': {', '.join(tickers)}...")
        print(f"Веса: {weights}")
        df = load_portfolio_data(tickers, years=5)
        df_features, scaler_features, feature_cols = prepare_features(df, tickers, weights)
        
        # Разделение данных
        X = df_features[feature_cols].values
        y = df_features['portfolio_return'].values
        
        # Нормализация целевой переменной
        y_mean, y_std = y.mean(), y.std()
        y_normalized = (y - y_mean) / y_std
        
        # Создание последовательностей для LSTM
        X_seq, y_seq = create_sequences(X, y_normalized, timesteps)
        
        # Разделение на тренировочные и тестовые данные
        train_size = int(0.8 * len(X_seq))
        X_train, X_test = X_seq[:train_size], X_seq[train_size:]
        y_train, y_test = y_seq[:train_size], y_seq[train_size:]
        y_test_actual = y[train_size + timesteps:]  # Определяем y_test_actual здесь
        
        # Обучение модели
        model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        model.fit(
            X_train, y_train,
            epochs=80,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
                ProgressCallback(callback)
            ]
        )
        
        # Прогноз на последний день
        latest_features = df_features[feature_cols].iloc[-timesteps:]  # Оставляем DataFrame
        latest_features = scaler_features.transform(latest_features)  # Теперь работает корректно
        latest_features = latest_features.reshape(1, timesteps, len(feature_cols))
        forecast_return_daily = model.predict(latest_features, verbose=0)[0][0] * y_std + y_mean
        forecast_return_daily_pct = forecast_return_daily * 100  # В %
        # Исправленная годовая доходность: компаундирование с динамическим ограничением
        forecast_return_annual = ((1 + forecast_return_daily_pct / 100) ** 252 - 1) * 100
        
        # Волатильность и просадка портфеля
        returns = df.pct_change().dropna()
        portfolio_returns = (returns * weights).sum(axis=1)
        volatility = portfolio_returns.rolling(20).std().iloc[-1] * np.sqrt(252) * 100  # В %
        cumulative_returns = (1 + portfolio_returns).cumprod()
        drawdowns = (cumulative_returns / cumulative_returns.cummax()) - 1
        max_drawdown = drawdowns.min() * 100  # В %
        
        # Ограничение на основе волатильности (например, не более 2x годовой волатильности)
        max_annual_return = volatility * 2  # Динамическое ограничение
        forecast_return_annual = np.clip(forecast_return_annual, -max_annual_return, max_annual_return)
        
        # Расчет MAE
        y_pred = model.predict(X_test, verbose=0).flatten()
        y_pred_actual = y_pred * y_std + y_mean
        mae_annual = np.mean(np.abs(y_test_actual - y_pred_actual))
        
        # Возвращаем результаты
        return {
            'yearly_return': forecast_return_annual,
            'daily_return': forecast_return_daily_pct,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'mae': mae_annual
        }
        
    except Exception as e:
        print(f"\n❌ Ошибка при анализе портфеля '{portfolio_name}': {str(e)}")
        raise