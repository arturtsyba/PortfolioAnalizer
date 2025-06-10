from PyQt5 import QtWidgets, QtCore
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from windows.ui_risk_window import Ui_RiskWindow
import NNClassification.NNClassification as NNClassification
import apimoex
import requests
import yfinance as yf
import mplcursors  # Для интерактивности диаграмм

class RiskWindowController(QtWidgets.QMainWindow):
    def __init__(self, db_ops, user_id, current_portfolio_id):
        super().__init__()
        self.ui = Ui_RiskWindow()
        self.ui.setupUi(self)
        self.db_ops = db_ops
        self.user_id = user_id
        self.current_portfolio_id = current_portfolio_id

        # Добавляем прогресс-бар программно
        self.progress_bar = QtWidgets.QProgressBar(self.ui.centralwidget)
        self.progress_bar.setGeometry(QtCore.QRect(20, 520, 400, 25))  # Настраиваем положение и размер
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)  # Скрываем по умолчанию

        # Настройка соединений
        self.setup_connections()
        self.load_portfolios()

    def setup_connections(self):
        """Подключение сигналов и слотов."""
        self.ui.predic_button.clicked.connect(self.calculate_risk)

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
            for index in range(self.ui.comboBox.count()):
                if self.ui.comboBox.itemData(index) == self.current_portfolio_id:
                    self.ui.comboBox.setCurrentIndex(index)
                    break

    def calculate_risk(self):
        """Обработка нажатия кнопки 'Рассчитать риск'."""
        # Получаем выбранный портфель из comboBox
        portfolio_id = self.ui.comboBox.itemData(self.ui.comboBox.currentIndex())
        if portfolio_id == 0:
            self.ui.final_predict_label.setText("Выберите портфель!")
            return

        # Получаем данные портфеля
        portfolio = self.db_ops.get_portfolio(portfolio_id)
        if not portfolio:
            self.ui.final_predict_label.setText("Портфель не найден!")
            return

        # Получаем данные активов
        assets = self.db_ops.get_assets(portfolio_id=portfolio_id)
        if not assets:
            self.ui.final_predict_label.setText("В портфеле нет активов!")
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
            print(f"Processing asset: {asset['ticker']}, type: {asset_type}, value: {value}")  # Отладка
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
            # Обновляем прогресс-бар
            self.progress_bar.setValue(i + 1)
            QtWidgets.QApplication.processEvents()  # Обновляем UI

        # Скрываем прогресс-бар после завершения загрузки
        self.progress_bar.setVisible(False)

        if total_value <= 0:
            self.ui.final_predict_label.setText("Общая стоимость портфеля равна 0 или отрицательна!")
            return

        # Нормализуем доли
        stocks = stocks / total_value if total_value else 0.0
        bonds = bonds / total_value if total_value else 0.0
        crypto = crypto / total_value if total_value else 0.0
        metals = metals / total_value if total_value else 0.0
        etf = etf / total_value if total_value else 0.0

        # Проверяем сумму долей и корректируем
        total_share = stocks + bonds + crypto + metals + etf
        print(f"Total share before correction: {total_share}, stocks: {stocks}, bonds: {bonds}, crypto: {crypto}, metals: {metals}, etf: {etf}")  # Отладка

        if total_share == 0:
            self.ui.final_predict_label.setText("Не удалось распределить активы по категориям!")
            return

        # Корректировка суммы долей до 1.0
        correction_factor = 1.0 / total_share
        stocks *= correction_factor
        bonds *= correction_factor
        crypto *= correction_factor
        metals *= correction_factor
        etf *= correction_factor

        total_share = stocks + bonds + crypto + metals + etf
        print(f"Total share after correction: {total_share}, stocks: {stocks}, bonds: {bonds}, crypto: {crypto}, metals: {metals}, etf: {etf}")  # Отладка

        # Получаем данные пользователя для горизонта
        user = self.db_ops.get_user_by_id(self.user_id)
        if not user:
            self.ui.final_predict_label.setText("Пользователь не найден!")
            return

        horizon = user.get("horizon")
        if horizon is None:
            self.ui.final_predict_label.setText("Горизонт инвестирования не указан! Пожалуйста, задайте его в окне целей.")
            return

        try:
            # Предсказание риска
            risk_label, probabilities = NNClassification.predict_risk(stocks, bonds, crypto, metals, etf, horizon)

            # Отладка: выводим значение risk_label
            print(f"Predicted risk label: {risk_label}")

            # Сбрасываем стиль перед установкой нового
            self.ui.final_predict_label.setStyleSheet("")

            # Установка цвета текста в зависимости от уровня риска
            if risk_label == "Низкий":
                self.ui.final_predict_label.setStyleSheet("color: #008000; font-weight: bold;")
            elif risk_label == "Умеренный":
                self.ui.final_predict_label.setStyleSheet("color: #ffd800; font-weight: bold;")
            elif risk_label == "Высокий":
                self.ui.final_predict_label.setStyleSheet("color: #8b0000; font-weight: bold;")
            else:
                self.ui.final_predict_label.setStyleSheet("color: black; font-weight: bold;")

            self.ui.final_predict_label.setText(risk_label)

            # Принудительное обновление виджета
            self.ui.final_predict_label.update()
            self.ui.final_predict_label.repaint()

            # Получаем размеры виджетов для диаграмм
            nn_view_width = self.ui.graphicsView_predictionNN.width()
            nn_view_height = self.ui.graphicsView_predictionNN.height()
            portfolio_view_width = self.ui.graphicsView_structureportfolio.width()
            portfolio_view_height = self.ui.graphicsView_structureportfolio.height()

            # Переводим размеры из пикселей в дюймы (примерно 100 пикселей = 1 дюйм для стандартного DPI)
            nn_fig_width = nn_view_width / 100
            nn_fig_height = nn_view_height / 100
            portfolio_fig_width = portfolio_view_width / 100
            portfolio_fig_height = portfolio_view_height / 100

            # Минимальный размер фигуры для предотвращения наложения текста
            min_fig_width = max(2.0, portfolio_fig_width)
            min_fig_height = max(2.0, portfolio_fig_height)

            # Визуализация круговой диаграммы (предсказание нейросети)
            labels = ['Низкий', 'Умеренный', 'Высокий']
            sizes = [prob * 100 for prob in probabilities]
            colors = ['#FF9999', '#66B2FF', '#99FF99']  # Различные цвета для темной и светлой тем
            fig, ax = plt.subplots(figsize=(nn_fig_width, nn_fig_height))
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')

            # Добавляем интерактивность с помощью mplcursors
            cursor = mplcursors.cursor(wedges, hover=True)
            cursor.connect("add", lambda sel: sel.annotation.set_text(
                f"{labels[sel.index]}: {sizes[sel.index]:.1f}%"))

            # Очистка и добавление canvas в graphicsView_predictionNN
            if self.ui.graphicsView_predictionNN.scene():
                self.ui.graphicsView_predictionNN.scene().clear()
            canvas = FigureCanvas(fig)
            scene = QtWidgets.QGraphicsScene()
            scene.addWidget(canvas)
            self.ui.graphicsView_predictionNN.setScene(scene)

            # Визуализация структуры портфеля
            portfolio_labels = ['Акции', 'Облигации', 'Криптовалюта', 'Металлы', 'ETF']
            portfolio_sizes = [stocks * 100, bonds * 100, crypto * 100, metals * 100, etf * 100]
            portfolio_colors = ['#FF6666', '#66CCCC', '#FFCC66', '#CC99FF', '#99FFCC']

            # Фильтруем только ненулевые доли
            active_labels = [label for label, size in zip(portfolio_labels, portfolio_sizes) if abs(size) > 1e-10]
            active_sizes = [size for size in portfolio_sizes if abs(size) > 1e-10]
            active_colors = [color for color, size in zip(portfolio_colors, portfolio_sizes) if abs(size) > 1e-10]

            fig2, ax2 = plt.subplots(figsize=(min_fig_width, min_fig_height))
            wedges2, texts2, autotexts2 = ax2.pie(active_sizes, labels=active_labels, colors=active_colors, autopct='%1.1f%%',
                                                  labeldistance=1.1, pctdistance=0.6, startangle=90)
            ax2.axis('equal')

            # Добавляем интерактивность с помощью mplcursors
            cursor2 = mplcursors.cursor(wedges2, hover=True)
            cursor2.connect("add", lambda sel: sel.annotation.set_text(
                f"{active_labels[sel.index]}: {active_sizes[sel.index]:.1f}%"))

            # Очистка и добавление canvas в graphicsView_structureportfolio
            if self.ui.graphicsView_structureportfolio.scene():
                self.ui.graphicsView_structureportfolio.scene().clear()
            canvas2 = FigureCanvas(fig2)
            scene2 = QtWidgets.QGraphicsScene()
            scene2.addWidget(canvas2)
            self.ui.graphicsView_structureportfolio.setScene(scene2)

            # Вычисление и отображение волатильности в процентах
            volatility = (stocks * 0.25 + crypto * 0.45 + bonds * 0.03 + metals * 0.15 + etf * 0.20) * 100
            self.ui.volat_label.setText(f"{volatility:.1f}%")

        except Exception as e:
            self.ui.final_predict_label.setText(f"Ошибка: {str(e)}")
            self.progress_bar.setVisible(False)

    def get_current_price(self, ticker, assets):
        """Получение текущей цены актива (адаптировано из MainApp)."""
        print(f"Fetching price for ticker: {ticker}")  # Для отладки
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
                            price = float(data[-1]['close'])
                            print(f"MOEX price for {ticker}: {price}")
                            return price
            except Exception as e:
                print(f"Error fetching MOEX price for {ticker}: {e}")
                for asset in assets:
                    if asset["ticker"] == ticker:
                        price = float(asset["purchase_price"])
                        print(f"Using purchase price for {ticker}: {price}")
                        return price
        else:
            try:
                stock = yf.Ticker(ticker)
                data = stock.history(period="1d")
                if not data.empty:
                    price = data["Close"].iloc[-1]
                    print(f"Yahoo Finance price for {ticker}: {price}")
                    return price
            except Exception as e:
                print(f"Error fetching Yahoo Finance price for {ticker}: {e}")
                for asset in assets:
                    if asset["ticker"] == ticker:
                        price = float(asset["purchase_price"])
                        print(f"Using purchase_price for {ticker}: {price}")
                        return price
        return 0.0

    def update_portfolio_id(self, portfolio_id):
        """Обновление текущего portfolio_id при изменении в главном окне."""
        self.current_portfolio_id = portfolio_id
        self.load_portfolios()