import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pickle
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QGraphicsScene, QProgressBar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import apimoex
import requests
import yfinance as yf

# Импортируем интерфейс
from windows.ui_optimization_window import Ui_OptimizationWindow

# Параметры активов и корреляционная матрица
ASSET_PARAMS = {
    'stocks': {'return': (0.08, 0.12), 'volatility': (0.20, 0.30)},
    'bonds': {'return': (0.02, 0.05), 'volatility': (0.01, 0.05)},
    'crypto': {'return': (0.15, 0.50), 'volatility': (0.30, 0.60)},
    'metals': {'return': (0.05, 0.10), 'volatility': (0.10, 0.20)},
    'etf': {'return': (0.06, 0.10), 'volatility': (0.10, 0.20)}
}
CORR_MATRIX = np.array([
    [1.0, 0.2, 0.7, 0.3, 0.4],
    [0.2, 1.0, 0.1, 0.4, 0.3],
    [0.7, 0.1, 1.0, 0.2, 0.2],
    [0.3, 0.4, 0.2, 1.0, 0.3],
    [0.4, 0.3, 0.2, 0.3, 1.0]
])

# Кастомный слой для нормализации весов
class NormalizeWeights(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(NormalizeWeights, self).__init__(**kwargs)

    def call(self, inputs):
        clipped = tf.clip_by_value(inputs, 0, 1)
        normalized = clipped / tf.reduce_sum(clipped, axis=-1, keepdims=True)
        return normalized

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return super(NormalizeWeights, self).get_config()

# Функция для вычисления волатильности и доходности
def calculate_portfolio_metrics(weights):
    returns = np.array([(ASSET_PARAMS[asset]['return'][0] + ASSET_PARAMS[asset]['return'][1]) / 2
                        for asset in ['stocks', 'bonds', 'crypto', 'metals', 'etf']])
    portfolio_return = np.sum(weights * returns)
    
    vols = np.array([(ASSET_PARAMS[asset]['volatility'][0] + ASSET_PARAMS[asset]['volatility'][1]) / 2
                     for asset in ['stocks', 'bonds', 'crypto', 'metals', 'etf']])
    portfolio_vol = np.sqrt(np.sum((weights * vols) ** 2) +
                            2 * np.sum([weights[i] * weights[j] * vols[i] * vols[j] * CORR_MATRIX[i, j]
                                       for i in range(5) for j in range(i + 1, 5)]))
    return portfolio_return, portfolio_vol

# Класс логики для окна оптимизации
class OptimizationWindowLogic(QtWidgets.QMainWindow):
    def __init__(self, db_ops, user_id, current_portfolio_id):
        super().__init__()
        self.ui = Ui_OptimizationWindow()
        self.ui.setupUi(self)
        self.db_ops = db_ops
        self.user_id = user_id
        self.current_portfolio_id = current_portfolio_id

        # Пути к модели и скейлеру
        self.MODEL_PATH = "NNOptimization/recommendation_model.keras"
        self.SCALER_PATH = "NNOptimization/recommendation_scaler.pkl"

        # Загрузка модели и скейлера
        self.model, self.scaler = self.load_model_and_scaler()

        # Добавляем прогресс-бар
        self.progress_bar = QProgressBar(self.ui.centralwidget)
        self.progress_bar.setGeometry(QtCore.QRect(20, 720, 400, 25))
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)

        # Подключаем кнопку "Оптимизировать веса"
        self.ui.predic_button.clicked.connect(self.optimize_weights)

        # Загружаем портфели в комбобокс
        self.load_portfolios()

        # Инициализация графиков с текущими данными портфеля
        self.load_portfolio_data()

        # Подключаем сигнал для перезагрузки при смене портфеля
        self.ui.comboBox.currentIndexChanged.connect(self.on_portfolio_changed)

    def load_model_and_scaler(self):
        try:
            model = tf.keras.models.load_model(self.MODEL_PATH, custom_objects={'NormalizeWeights': NormalizeWeights})
            with open(self.SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
            print("Модель и скейлер успешно загружены.")
            return model, scaler
        except Exception as e:
            print(f"Ошибка при загрузке модели или скейлера: {e}")
            return None, None

    def load_portfolios(self):
        """Загрузка списка портфелей в QComboBox."""
        portfolios = self.db_ops.get_portfolios(user_id=self.user_id)
        self.ui.comboBox.clear()
        if not portfolios:
            self.ui.comboBox.addItem("Нет портфелей", 0)
            self.current_portfolio_id = None
        else:
            for portfolio in portfolios:
                self.ui.comboBox.addItem(portfolio["portfolio_name"], portfolio["portfolio_id"])
            # Устанавливаем текущий портфель, если он есть
            if self.current_portfolio_id:
                for index in range(self.ui.comboBox.count()):
                    if self.ui.comboBox.itemData(index) == self.current_portfolio_id:
                        self.ui.comboBox.setCurrentIndex(index)
                        break

    def on_portfolio_changed(self, index):
        """Обработка смены выбранного портфеля."""
        self.current_portfolio_id = self.ui.comboBox.itemData(index)
        self.load_portfolio_data()

    def load_portfolio_data(self):
        """Загрузка данных портфеля из базы данных и отображение текущей структуры."""
        if not self.current_portfolio_id or self.current_portfolio_id == 0:
            self.ui.final_predict_label.setText("Выберите портфель!")
            if self.ui.graphicsView_structureportfolio.scene():
                self.ui.graphicsView_structureportfolio.scene().clear()
            return

        # Получаем данные портфеля
        portfolio = self.db_ops.get_portfolio(self.current_portfolio_id)
        if not portfolio:
            self.ui.final_predict_label.setText("Портфель не найден!")
            if self.ui.graphicsView_structureportfolio.scene():
                self.ui.graphicsView_structureportfolio.scene().clear()
            return

        # Получаем данные активов
        assets = self.db_ops.get_assets(portfolio_id=self.current_portfolio_id)
        if not assets:
            self.ui.final_predict_label.setText("В портфеле нет активов!")
            if self.ui.graphicsView_structureportfolio.scene():
                self.ui.graphicsView_structureportfolio.scene().clear()
            return

        # Показываем прогресс-бар
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(assets))
        self.progress_bar.setValue(0)

        # Вычисляем доли активов
        stocks, bonds, crypto, metals, etf = 0.0, 0.0, 0.0, 0.0, 0.0
        total_value = 0.0
        for i, asset in enumerate(assets):
            count = float(asset["count_assets"])
            current_price = self.get_current_price(asset["ticker"], assets)
            value = count * current_price
            total_value += value
            asset_type = asset["asset_type"]
            print(f"Processing asset: {asset['ticker']}, type: {asset_type}, value: {value}")
            if asset_type == "Акции":
                stocks += value
            elif asset_type == "Облигации":
                bonds += value
            elif asset_type == "Криптовалюты":
                crypto += value
            elif asset_type == "Металлы":
                metals += value
            elif asset_type == "ETF":
                etf += value
            else:
                print(f"Unknown asset type: {asset_type}")
            self.progress_bar.setValue(i + 1)
            QtWidgets.QApplication.processEvents()

        # Скрываем прогресс-бар после завершения загрузки
        self.progress_bar.setVisible(False)

        if total_value <= 0:
            self.ui.final_predict_label.setText("Общая стоимость портфеля равна 0 или отрицательна!")
            if self.ui.graphicsView_structureportfolio.scene():
                self.ui.graphicsView_structureportfolio.scene().clear()
            return

        # Нормализуем доли
        stocks = stocks / total_value if total_value else 0.0
        bonds = bonds / total_value if total_value else 0.0
        crypto = crypto / total_value if total_value else 0.0
        metals = metals / total_value if total_value else 0.0
        etf = etf / total_value if total_value else 0.0

        # Проверяем сумму долей и корректируем
        total_share = stocks + bonds + crypto + metals + etf
        print(f"Total share before correction: {total_share}, stocks: {stocks}, bonds: {bonds}, crypto: {crypto}, metals: {metals}, etf: {etf}")

        if total_share == 0:
            self.ui.final_predict_label.setText("Не удалось распределить активы по категориям!")
            if self.ui.graphicsView_structureportfolio.scene():
                self.ui.graphicsView_structureportfolio.scene().clear()
            return

        # Корректировка суммы долей до 1.0
        correction_factor = 1.0 / total_share
        stocks *= correction_factor
        bonds *= correction_factor
        crypto *= correction_factor
        metals *= correction_factor
        etf *= correction_factor

        total_share = stocks + bonds + crypto + metals + etf
        print(f"Total share after correction: {total_share}, stocks: {stocks}, bonds: {bonds}, crypto: {crypto}, metals: {metals}, etf: {etf}")

        # Сохраняем текущие веса
        self.current_weights = [stocks, bonds, crypto, metals, etf]

        # Отображаем текущую структуру портфеля
        self.plot_pie_chart(self.current_weights, self.ui.graphicsView_structureportfolio, "Текущий портфель")

    def get_current_price(self, ticker, assets):
        """Получение текущей цены актива."""
        if ticker.endswith('-ru'):
            try:
                moex_ticker = ticker.replace('-ru', '')
                with requests.Session() as session:
                    for board in ['TQBR', 'TQPD']:
                        data = apimoex.get_board_candles(
                            session=session,
                            security=moex_ticker,
                            board=board,
                            interval=24,
                            limit=1
                        )
                        if data and len(data) > 0:
                            return float(data[-1]['close'])
            except Exception as e:
                print(f"Error fetching MOEX price for {ticker}: {e}")
                for asset in assets:
                    if asset["ticker"] == ticker:
                        return float(asset["purchase_price"])
        else:
            try:
                stock = yf.Ticker(ticker)
                data = stock.history(period="1d")
                if not data.empty:
                    return data["Close"].iloc[-1]
            except Exception as e:
                print(f"Error fetching Yahoo Finance price for {ticker}: {e}")
                for asset in assets:
                    if asset["ticker"] == ticker:
                        return float(asset["purchase_price"])
        return 0.0

    def plot_pie_chart(self, weights, graphics_view, title):
        portfolio_labels = ['Акции', 'Облигации', 'Криптовалюта', 'Металлы', 'ETF']
        portfolio_sizes = [w * 100 for w in weights]  # Преобразуем в проценты
        portfolio_colors = ['#FF6666', '#66CCCC', '#FFCC66', '#CC99FF', '#99FFCC']

        # Фильтруем только ненулевые доли
        active_labels = [label for label, size in zip(portfolio_labels, portfolio_sizes) if abs(size) > 1e-10]
        active_sizes = [size for size in portfolio_sizes if abs(size) > 1e-10]
        active_colors = [color for color, size in zip(portfolio_colors, portfolio_sizes) if abs(size) > 1e-10]

        # Получаем размеры виджета
        viewport_size = graphics_view.viewport().size()
        view_width = max(7.0, viewport_size.width() / 60)  # Увеличен минимальный размер и уменьшен коэффициент
        view_height = max(7.0, viewport_size.height() / 60)

        # Создаем фигуру с увеличенным размером
        fig, ax = plt.subplots(figsize=(view_width, view_height), dpi=150)
        wedges, texts, autotexts = ax.pie(active_sizes, labels=active_labels, colors=active_colors,
                                        autopct='%1.1f%%', labeldistance=1.1, pctdistance=0.6, startangle=90)

        # Увеличение шрифта
        plt.setp(texts, size=18)  # Размер шрифта для меток
        plt.setp(autotexts, size=18)  # Размер шрифта для процентов
        ax.set_title(title, fontsize=20)  # Размер шрифта для заголовка

        ax.axis('equal')

        # Очистка и добавление canvas в graphicsView
        if graphics_view.scene():
            graphics_view.scene().clear()
        canvas = FigureCanvas(fig)
        scene = QGraphicsScene()
        scene.addWidget(canvas)
        graphics_view.setScene(scene)
        graphics_view.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
        graphics_view.setRenderHint(QtGui.QPainter.Antialiasing)

        # Принудительное обновление размеров
        self.ui.graphicsView_structureportfolio.updateGeometry()
        QtCore.QTimer.singleShot(100, lambda: graphics_view.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio))

    def optimize_weights(self):
        if self.model is None or self.scaler is None:
            self.ui.final_predict_label.setText("Ошибка: модель или скейлер не загружены.")
            return

        # Проверяем, что текущие веса были загружены
        if not hasattr(self, 'current_weights'):
            self.ui.final_predict_label.setText("Ошибка: данные портфеля не загружены.")
            return

        # Получаем данные пользователя для горизонта и цели
        user = self.db_ops.get_user_by_id(self.user_id)
        if not user:
            self.ui.final_predict_label.setText("Пользователь не найден!")
            return

        horizon = user.get("horizon")
        goal_str = user.get("investment_goal")
        if horizon is None or goal_str is None:
            self.ui.final_predict_label.setText("Горизонт или цель инвестирования не указаны! Пожалуйста, задайте их в окне целей.")
            return

        # Преобразуем строковое значение investment_goal в число
        goal_mapping = {
            'Сохранение капитала': 0,
            'Рост капитала': 1,
            'Сбалансированный': 2
        }
        goal = goal_mapping.get(goal_str)
        if goal is None:
            self.ui.final_predict_label.setText(f"Неизвестная цель инвестирования: {goal_str}. Ожидаются: 'Сохранение капитала', 'Рост капитала', 'Сбалансированный'.")
            return

        weights = np.array(self.current_weights)

        # Вычисляем метрики портфеля
        portfolio_return, volatility = calculate_portfolio_metrics(weights)

        # Подготавливаем входные данные для модели
        input_data = np.zeros(20)  # 20 признаков, как в тренировочных данных
        input_data[:5] = weights
        input_data[5] = volatility
        input_data[6] = portfolio_return
        input_data[7 + goal] = 1.0  # investment_goal (one-hot: 7, 8, 9)
        input_data[10 + horizon - 1] = 1.0  # horizon (one-hot: 10-19)

        # Предсказание
        input_data_scaled = self.scaler.transform(input_data.reshape(1, -1))
        prediction = self.model.predict(input_data_scaled, verbose=0)[0]
        prediction = np.clip(prediction, 0, 1)
        prediction /= prediction.sum()

        # Обновляем диаграмму оптимизированных весов
        self.plot_pie_chart(prediction, self.ui.graphicsView_predictionNN, "Оптимизированный портфель")

        # Формируем рекомендации в процентах
        assets = ['акций', 'облигаций', 'криптовалюты', 'металлов', 'ETF']
        current_weights = [w * 100 for w in weights]  # Конвертируем в проценты
        prediction_percent = [p * 100 for p in prediction]  # Конвертируем в проценты
        recommendation_text = ""
        for i, (current, predicted, asset) in enumerate(zip(current_weights, prediction_percent, assets)):
            change = predicted - current
            action = "Увеличить" if change > 0 else "Уменьшить"
            if abs(change) < 0.1:  # Если изменение незначительное, пропускаем
                continue
            if change > 0:
                recommendation_text += f"{action} долю {asset} до {predicted:.1f}%\n"
            else:
                recommendation_text += f"{action} долю {asset} на {abs(change):.1f}%\n"

        self.ui.final_predict_label.setText(recommendation_text)

    def closeEvent(self, event):
        plt.close('all')  # Закрываем все окна matplotlib при закрытии
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    # Для теста нужно передать db_ops, user_id и current_portfolio_id
    from data.db_operations import DBOperations
    db_ops = DBOperations()
    window = OptimizationWindowLogic(db_ops=db_ops, user_id=1, current_portfolio_id=1)
    window.show()
    sys.exit(app.exec_())