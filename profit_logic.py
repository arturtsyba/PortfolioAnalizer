import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QGraphicsScene, QApplication
from PyQt5.QtChart import QChart, QChartView, QLineSeries, QValueAxis, QDateTimeAxis
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtCore import Qt, QDateTime
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

# Импортируем интерфейс
from windows.ui_profitble import Ui_ProfitbleWindow
from data.db_operations import DBOperations
from NNProfitble import portfolio_forecast

class ProfitbleWindowLogic(QtWidgets.QMainWindow):
    def __init__(self, db_ops, user_id, current_portfolio_id):
        super().__init__()
        self.ui = Ui_ProfitbleWindow()
        self.ui.setupUi(self)
        self.db_ops = db_ops
        self.user_id = user_id
        self.current_portfolio_id = current_portfolio_id

        # Выравниваем текст над прогресс-баром по центру
        self.ui.label_3.setAlignment(QtCore.Qt.AlignCenter)

        # Подключаем кнопку "Прогноз"
        self.ui.predic_button.clicked.connect(self.predict_portfolio)

        # Загружаем портфели в комбобокс
        self.load_portfolios()

        # Инициализация данных портфеля
        self.load_portfolio_data()

        # Подключаем сигнал для перезагрузки при смене портфеля
        self.ui.comboBox.currentIndexChanged.connect(self.on_portfolio_changed)

    def showEvent(self, event):
        """Вызывается при каждом открытии окна."""
        super().showEvent(event)
        # Перезагружаем портфели и данные при открытии окна
        self.load_portfolios()
        self.load_portfolio_data()

    def load_portfolios(self):
        """Загрузка списка портфелей в QComboBox."""
        portfolios = self.db_ops.get_portfolios(user_id=self.user_id)
        print("Portfolios loaded:", portfolios)
        self.ui.comboBox.clear()
        if not portfolios:
            self.ui.comboBox.addItem("Нет портфелей", 0)
            self.current_portfolio_id = None
            print("No portfolios found for user_id:", self.user_id)
        else:
            for portfolio in portfolios:
                self.ui.comboBox.addItem(portfolio["portfolio_name"], portfolio["portfolio_id"])
            if self.current_portfolio_id:
                for index in range(self.ui.comboBox.count()):
                    if self.ui.comboBox.itemData(index) == self.current_portfolio_id:
                        self.ui.comboBox.setCurrentIndex(index)
                        break
            else:
                self.current_portfolio_id = self.ui.comboBox.itemData(0)
            print("Current portfolio ID set to:", self.current_portfolio_id)

    def on_portfolio_changed(self, index):
        """Обработка смены выбранного портфеля."""
        self.current_portfolio_id = self.ui.comboBox.itemData(index)
        print("Portfolio changed to ID:", self.current_portfolio_id)
        self.load_portfolio_data()

    def load_portfolio_data(self):
        """Загрузка данных портфеля из базы данных и отображение текущей структуры."""
        if not self.current_portfolio_id or self.current_portfolio_id == 0:
            self.ui.label_3.setText("Выберите портфель!")
            if self.ui.graphicsView_structureportfolio.scene():
                self.ui.graphicsView_structureportfolio.scene().clear()
            if self.ui.graphicsView.scene():
                self.ui.graphicsView.scene().clear()
            return

        portfolio = self.db_ops.get_portfolio(self.current_portfolio_id)
        if not portfolio:
            self.ui.label_3.setText("Портфель не найден!")
            if self.ui.graphicsView_structureportfolio.scene():
                self.ui.graphicsView_structureportfolio.scene().clear()
            if self.ui.graphicsView.scene():
                self.ui.graphicsView.scene().clear()
            return

        assets = self.db_ops.get_assets(portfolio_id=self.current_portfolio_id)
        if not assets:
            self.ui.label_3.setText("В портфеле нет активов!")
            if self.ui.graphicsView_structureportfolio.scene():
                self.ui.graphicsView_structureportfolio.scene().clear()
            if self.ui.graphicsView.scene():
                self.ui.graphicsView.scene().clear()
            return

        total_value = 0.0
        weights = []
        tickers = []
        for asset in assets:
            count = float(asset["count_assets"])
            current_price = self.get_current_price(asset["ticker"], assets)
            value = count * current_price
            total_value += value
            weights.append(value)
            tickers.append(asset["ticker"])
            QtWidgets.QApplication.processEvents()

        if total_value <= 0:
            self.ui.label_3.setText("Общая стоимость портфеля равна 0 или отрицательна!")
            if self.ui.graphicsView_structureportfolio.scene():
                self.ui.graphicsView_structureportfolio.scene().clear()
            if self.ui.graphicsView.scene():
                self.ui.graphicsView.scene().clear()
            return

        weights = np.array(weights) / total_value
        self.current_weights = weights
        self.current_tickers = tickers
        self.current_portfolio_name = portfolio["portfolio_name"]

        self.plot_pie_chart(self.current_weights, self.current_tickers, self.ui.graphicsView_structureportfolio, "Структура портфеля")

        self.ui.yearprofit_label.setText("Пока пусто")
        self.ui.dayprofit_label.setText("Пока пусто")
        self.ui.volatility_label.setText("Пока пусто")
        self.ui.maxprosadka_label.setText("Пока пусто")

        if self.ui.graphicsView.scene():
            self.ui.graphicsView.scene().clear()

    def get_current_price(self, ticker, assets):
        """Получение текущей цены актива."""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period="1d")
            if not data.empty:
                return data["Close"].iloc[-1]
        except Exception as e:
            print(f"Ошибка при получении цены для {ticker}: {e}")
            for asset in assets:
                if asset["ticker"] == ticker:
                    return float(asset["purchase_price"])
        return 0.0

    def load_historical_prices(self, ticker, start_date, end_date):
        """Загрузка исторических цен актива с помощью yfinance."""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date, auto_adjust=True)
            if data.empty:
                print(f"No data found for {ticker} on Yahoo Finance")
                return None
            data.index = data.index.tz_localize(None)
            return data[['Close']]
        except Exception as e:
            print(f"Error fetching data for {ticker} on Yahoo Finance: {e}")
            return None

    def calculate_portfolio_metrics(self, assets, progress_dialog=None):
        """Расчёт метрик портфеля и активов на основе исторических цен."""
        if not assets:
            return None, None, [0.0] * 5, [], (0.0, 0.0)

        purchase_dates = [datetime.strptime(asset["purchase_date"], "%Y-%m-%d") for asset in assets]
        start_date = max(min(purchase_dates), datetime(2017, 1, 1))
        end_date = datetime.now()

        portfolio_values = {}
        date_list = []
        total_weight = 0.0

        total_steps = len(assets) + 1
        if progress_dialog:
            progress_dialog.setMaximum(total_steps)
            progress_dialog.setValue(0)

        for idx, asset in enumerate(assets):
            if progress_dialog:
                progress_dialog.setLabelText(f"Загрузка данных для {asset['ticker']}...")
                progress_dialog.setValue(idx)
                QApplication.processEvents()

            ticker = asset["ticker"]
            count = float(asset["count_assets"])
            purchase_price = float(asset["purchase_price"])
            purchase_date = datetime.strptime(asset["purchase_date"], "%Y-%m-%d")

            data = self.load_historical_prices(ticker, start_date, end_date)
            if data is None or data.empty:
                print(f"Skipping {ticker} due to missing data")
                continue

            asset_value = count * purchase_price
            total_weight += asset_value

            for date, row in data.iterrows():
                date = date.to_pydatetime()
                if date < purchase_date:
                    continue
                price = row["Close"]
                value = count * price

                if date in portfolio_values:
                    portfolio_values[date] += value
                else:
                    portfolio_values[date] = value
                    date_list.append(date)

        date_list.sort()
        values = [portfolio_values.get(date, 0) for date in date_list]

        # Сглаживание значений: ограничиваем изменения между соседними днями
        smoothed_values = []
        for i in range(len(values)):
            if i == 0:
                smoothed_values.append(values[i])
            else:
                prev_value = smoothed_values[-1]
                change = (values[i] - prev_value) / prev_value if prev_value != 0 else 0
                if abs(change) > 0.1:
                    smoothed_values.append(prev_value * (1 + 0.1 if change > 0 else 0.9))
                else:
                    smoothed_values.append(values[i])

        daily_returns = np.array([])
        if len(smoothed_values) > 1:
            daily_returns = np.array([(smoothed_values[i] - smoothed_values[i-1]) / smoothed_values[i-1] * 100 for i in range(1, len(smoothed_values))])
            volatility = np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 0 else 0.0
            mean_return = np.mean(daily_returns) * 252 if len(daily_returns) > 0 else 0.0
        else:
            volatility = 0.0
            mean_return = 0.0

        total_return = 0.0
        for asset in assets:
            ticker = asset["ticker"]
            count = float(asset["count_assets"])
            purchase_price = float(asset["purchase_price"])
            data = self.load_historical_prices(ticker, start_date, end_date)
            if data is not None and not data.empty:
                last_price = data["Close"].iloc[-1]
                asset_return = (last_price - purchase_price) / purchase_price * 100
                asset_value = count * purchase_price
                total_return += asset_return * asset_value
        portfolio_return = total_return / total_weight if total_weight > 0 else 0.0

        risk_free_rate = 2.0
        sharpe_ratio = (portfolio_return - risk_free_rate) / volatility if volatility > 0 else 0.0

        return date_list, smoothed_values, [portfolio_return, volatility, 0.0, 0.0, sharpe_ratio, mean_return], [], (0.0, 0.0)

    def plot_portfolio_value(self, dates, values, yearly_return):
        """Отрисовка графика движения цены портфеля без прогнозной линии."""
        print("Попытка построить график...")
        print(f"Даты: {dates[:5] if dates else None}...")
        print(f"Значения: {values[:5] if values else None}...")
        print(f"Yearly return: {yearly_return}")

        if not dates or not values or len(dates) == 0 or len(values) == 0:
            print("Невозможно построить график: данные отсутствуют")
            self.ui.label_3.setText("Не удалось построить график: данные отсутствуют")
            self.plot_test_graph()
            return

        # Серия только для исторических данных
        historical_series = QLineSeries()
        for date, value in zip(dates, values):
            qdate = QDateTime(date)
            historical_series.append(qdate.toMSecsSinceEpoch(), value)

        # Создание графика
        chart = QChart()
        chart.addSeries(historical_series)
        
        # Уменьшенный шрифт заголовка
        title_font = QtGui.QFont()
        title_font.setPointSize(14)
        chart.setTitleFont(title_font)
        chart.setTitle("Движение стоимости портфеля")

        historical_series.setName("Историческая стоимость")

        # Ось X (даты) с уменьшенным шрифтом
        axis_x = QDateTimeAxis()
        axis_x.setFormat("dd-MM-yyyy")
        axis_x_font = QtGui.QFont()
        axis_x_font.setPointSize(8)
        axis_x.setLabelsFont(axis_x_font)
        axis_x.setTitleFont(axis_x_font)
        axis_x.setTitleText("Дата")
        chart.addAxis(axis_x, Qt.AlignBottom)
        historical_series.attachAxis(axis_x)

        # Ось Y (стоимость) с уменьшенным шрифтом
        axis_y = QValueAxis()
        axis_y.setLabelFormat("%.2f")
        axis_y_font = QtGui.QFont()
        axis_y_font.setPointSize(8)
        axis_y.setLabelsFont(axis_y_font)
        axis_y.setTitleFont(axis_y_font)
        axis_y.setTitleText("Стоимость ($)")
        
        # Настройка диапазона только для исторических данных
        if values:
            lower_bound = np.percentile(values, 5)
            upper_bound = np.percentile(values, 95)
            range_span = upper_bound - lower_bound
            min_value = lower_bound - 0.1 * range_span
            max_value = upper_bound + 0.1 * range_span
            min_value = max(min_value, 0)
        else:
            min_value = 0.0
            max_value = 100.0
        axis_y.setRange(min_value, max_value)
        chart.addAxis(axis_y, Qt.AlignLeft)
        historical_series.attachAxis(axis_y)

        # Уменьшенный шрифт легенды
        legend = chart.legend()
        legend_font = QtGui.QFont()
        legend_font.setPointSize(10)
        legend.setFont(legend_font)
        legend.setVisible(True)
        legend.setAlignment(Qt.AlignBottom)

        # Уменьшенные отступы
        chart.setMargins(QtCore.QMargins(10, 5, 5, 5))

        # Отображение в graphicsView с увеличенным размером
        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.Antialiasing)
        view_width = self.ui.graphicsView.width()
        view_height = self.ui.graphicsView.height()
        
        # Увеличиваем размер графика до почти полного размера виджета
        chart_view.setFixedSize(int(view_width * 0.98), int(view_height * 0.95))
        
        scene = QGraphicsScene(0, 0, view_width, view_height)
        scene.addWidget(chart_view)
        self.ui.graphicsView.setScene(scene)
        self.ui.graphicsView.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
        print("График должен быть отображён")

    def plot_test_graph(self):
        """Построение тестового графика для проверки graphicsView."""
        print("Попытка построить тестовый график...")
        view_width = self.ui.graphicsView.width()
        view_height = self.ui.graphicsView.height()

        series = QLineSeries()
        x = np.linspace(0, 10, 100)
        for i in x:
            qdate = QDateTime(datetime.now()).addMSecs(int(i * 1000))
            series.append(qdate.toMSecsSinceEpoch(), np.sin(i))

        chart = QChart()
        chart.addSeries(series)
        
        # Увеличение шрифта заголовка
        title_font = QtGui.QFont()
        title_font.setPointSize(26)
        chart.setTitleFont(title_font)
        chart.setTitle("Тестовый график (sin)")

        series.setName("Тестовая линия")

        # Ось X (даты)
        axis_x = QDateTimeAxis()
        axis_x.setFormat("dd-MM-yyyy")
        axis_x_font = QtGui.QFont()
        axis_x_font.setPointSize(12)
        axis_x.setLabelsFont(axis_x_font)
        axis_x.setTitleFont(axis_x_font)
        axis_x.setTitleText("Время")
        chart.addAxis(axis_x, Qt.AlignBottom)
        series.attachAxis(axis_x)

        # Ось Y (значения)
        axis_y = QValueAxis()
        axis_y.setLabelFormat("%.2f")
        axis_y_font = QtGui.QFont()
        axis_y_font.setPointSize(12)
        axis_y.setLabelsFont(axis_y_font)
        axis_y.setTitleFont(axis_y_font)
        axis_y.setTitleText("Значение")
        chart.addAxis(axis_y, Qt.AlignLeft)
        series.attachAxis(axis_y)

        # Увеличение шрифта легенды
        legend = chart.legend()
        legend_font = QtGui.QFont()
        legend_font.setPointSize(12)
        legend.setFont(legend_font)

        # Добавляем отступы
        chart.setMargins(QtCore.QMargins(30, 10, 10, 10))

        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.Antialiasing)
        
        # ИСПРАВЛЕНИЕ: Приведение к int для setFixedSize
        chart_view.setFixedSize(int(view_width * 0.9), int(view_height * 0.9))
        
        scene = QGraphicsScene(0, 0, view_width, view_height)
        scene.addWidget(chart_view)
        self.ui.graphicsView.setScene(scene)
        self.ui.graphicsView.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
        print("Тестовый график должен быть отображён")

    def plot_pie_chart(self, weights, labels, graphics_view, title):
        """Отображение кругового графика весов."""
        sizes = [w * 100 for w in weights]
        colors = ['#FF6666', '#66CCCC', '#FFCC66', '#CC99FF', '#99FFCC', '#FF99CC', '#66FF99']

        active_labels = [label for label, size in zip(labels, sizes) if abs(size) > 1e-10]
        active_sizes = [size for size in sizes if abs(size) > 1e-10]
        active_colors = [colors[i % len(colors)] for i, size in enumerate(sizes) if abs(size) > 1e-10]

        viewport_size = graphics_view.viewport().size()
        view_width = max(7.0, viewport_size.width() / 70)
        view_height = max(7.0, viewport_size.height() / 70)

        fig, ax = plt.subplots(figsize=(view_width, view_height), dpi=150)
        wedges, texts, autotexts = ax.pie(active_sizes, labels=active_labels, colors=active_colors,
                                          autopct='%1.1f%%', labeldistance=1.1, pctdistance=0.6, startangle=90)

        plt.setp(texts, size=19)
        plt.setp(autotexts, size=19)
        ax.set_title(title, fontsize=22)

        ax.axis('equal')
        plt.tight_layout()

        if graphics_view.scene():
            graphics_view.scene().clear()
        canvas = FigureCanvas(fig)
        scene = QGraphicsScene()
        scene.addWidget(canvas)
        graphics_view.setScene(scene)
        graphics_view.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
        graphics_view.setRenderHint(QtGui.QPainter.Antialiasing)

        self.ui.graphicsView_structureportfolio.updateGeometry()
        QtCore.QTimer.singleShot(100, lambda: graphics_view.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio))

    def predict_portfolio(self):
        """Запуск прогнозирования для выбранного портфеля."""
        if not self.current_portfolio_id or self.current_portfolio_id == 0:
            self.ui.label_3.setText("Выберите портфель!")
            return

        if not hasattr(self, 'current_weights') or not hasattr(self, 'current_tickers'):
            self.ui.label_3.setText("Данные портфеля не загружены!")
            return

        self.ui.progressBar.setVisible(True)
        self.ui.progressBar.setRange(0, 80)
        self.ui.progressBar.setValue(0)
        self.ui.label_3.setText("Идёт прогнозирование, подождите пожалуйста")
        self.ui.statusbar.showMessage("Обучение...")

        def update_progress(current, total):
            progress = (current / total) * 100
            self.ui.progressBar.setValue(int(progress))
            QtWidgets.QApplication.processEvents()

        assets = self.db_ops.get_assets(portfolio_id=self.current_portfolio_id)
        try:
            result = portfolio_forecast.analyze_portfolio(
                tickers=self.current_tickers,
                weights=self.current_weights,
                portfolio_name=self.current_portfolio_name,
                callback=update_progress
            )

            self.ui.yearprofit_label.setText(f"{result['yearly_return']:.2f}%")
            self.ui.dayprofit_label.setText(f"{result['daily_return']:.2f}%")
            self.ui.volatility_label.setText(f"{result['volatility']:.2f}%")
            self.ui.maxprosadka_label.setText(f"{result['max_drawdown']:.2f}%")

            dates, values, _, _, _ = self.calculate_portfolio_metrics(assets)
            if dates and values:
                self.plot_portfolio_value(dates, values, result['yearly_return'])
            else:
                self.ui.label_3.setText("Не удалось загрузить данные для графика")

            self.ui.label_3.setText("Прогнозирование завершено")
            self.ui.statusbar.showMessage("Готово")
        except Exception as e:
            print(f"Ошибка при прогнозировании: {e}")
            self.ui.label_3.setText(f"Ошибка: {str(e)}")
            self.ui.statusbar.showMessage("Ошибка при прогнозировании")
        finally:
            self.ui.progressBar.setVisible(False)

    def closeEvent(self, event):
        """Закрытие окна и очистка."""
        plt.close('all')
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    db_ops = DBOperations()
    window = ProfitbleWindowLogic(db_ops=db_ops, user_id=1, current_portfolio_id=1)
    window.show()
    sys.exit(app.exec_())