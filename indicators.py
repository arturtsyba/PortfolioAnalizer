from PyQt5.QtWidgets import QMainWindow, QTableWidgetItem, QHeaderView, QGraphicsScene, QProgressDialog, QApplication, QGraphicsRectItem, QGraphicsTextItem
from PyQt5.QtChart import QChart, QChartView, QLineSeries, QScatterSeries, QValueAxis, QDateTimeAxis, QLogValueAxis
from PyQt5.QtGui import QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QDateTime
from windows.ui_indicators import Ui_indicatorwindow
from data.db_operations import DBOperations
import yfinance as yf
import apimoex
import requests
from datetime import datetime
import numpy as np
import pandas as pd
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IndicatorsWindow(QMainWindow):
    def __init__(self, user_id, portfolio_id):
        super().__init__()
        self.ui = Ui_indicatorwindow()
        self.ui.setupUi(self)
        self.user_id = user_id
        self.portfolio_id = portfolio_id
        self.db_ops = DBOperations()
        self.connect_signals()
        self.load_data()

    def connect_signals(self):
        """Подключение сигналов для кнопок."""
        try:
            self.ui.pushButton_1.clicked.connect(self.go_to_portfolio)
        except AttributeError:
            logger.error("pushButton_1 not found in UI. Ensure it exists in indicators_window.ui")

    def go_to_portfolio(self):
        """Закрытие окна индикаторов."""
        self.close()

    def load_historical_prices(self, ticker, start_date, end_date):
        """Загрузка исторических цен актива с помощью yfinance или apimoex."""
        if start_date >= end_date:
            logger.error(f"Invalid date range for {ticker}: start_date ({start_date}) >= end_date ({end_date})")
            return None

        if ticker.endswith('-ru'):
            try:
                moex_ticker = ticker.replace('-ru', '')
                with requests.Session() as session:
                    data = apimoex.get_board_history(
                        session=session,
                        security=moex_ticker,
                        board='TQBR',
                        start=start_date.strftime('%Y-%m-%d'),
                        end=end_date.strftime('%Y-%m-%d')
                    )
                    if data:
                        df = pd.DataFrame(data)
                        df['date'] = pd.to_datetime(df['TRADEDATE'])
                        df.set_index('date', inplace=True)
                        df.rename(columns={'CLOSE': 'Close'}, inplace=True)
                        df.index = df.index.tz_localize(None)
                        logger.info(f"Loaded {len(df)} rows for {ticker} from MOEX")
                        return df[['Close']]
                    logger.warning(f"No data found for {ticker} on MOEX")
                    return None
            except Exception as e:
                logger.error(f"Error fetching data for {ticker} on MOEX: {e}")
                return None
        else:
            try:
                stock = yf.Ticker(ticker)
                data = stock.history(start=start_date, end=end_date)
                if data.empty:
                    logger.warning(f"No data found for {ticker} on Yahoo Finance")
                    return None
                data.index = data.index.tz_localize(None)
                logger.info(f"Loaded {len(data)} rows for {ticker} from Yahoo Finance")
                return data
            except Exception as e:
                logger.error(f"Error fetching data for {ticker} on Yahoo Finance: {e}")
                return None

    def calculate_asset_metrics(self, ticker, purchase_date, start_date, end_date):
        """Расчёт доходности и волатильности для отдельного актива."""
        data = self.load_historical_prices(ticker, start_date, end_date)
        if data is None or data.empty:
            logger.warning(f"No historical data for {ticker}, returning zero metrics")
            return 0.0, 0.0

        purchase_date = datetime.strptime(purchase_date, "%Y-%m-%d")
        data = data[data.index >= purchase_date]

        if data.empty:
            logger.warning(f"No data for {ticker} after purchase date {purchase_date}")
            return 0.0, 0.0

        first_price = data["Close"].iloc[0]
        last_price = data["Close"].iloc[-1]

        if first_price <= 0 or last_price <= 0:
            logger.warning(f"Invalid prices for {ticker}: first_price={first_price}, last_price={last_price}")
            return 0.0, 0.0

        asset_return = (last_price - first_price) / first_price * 100

        daily_returns = data["Close"].pct_change().dropna() * 100
        if len(daily_returns) > 1:
            volatility = np.std(daily_returns) * np.sqrt(252)
        else:
            volatility = 0.0

        return asset_return, volatility

    def calculate_portfolio_metrics(self, assets, progress_dialog=None):
        """Расчёт метрик портфеля и активов на основе исторических цен."""
        if not assets:
            logger.warning("No assets provided for portfolio metrics calculation")
            return None, None, [0.0] * 6, [], (0.0, 0.0)

        purchase_dates = [datetime.strptime(asset["purchase_date"], "%Y-%m-%d") for asset in assets]
        start_date = min(purchase_dates)
        end_date = datetime.now()

        portfolio_values = {}
        asset_data = {}
        date_list = []
        total_weight = 0.0
        asset_metrics = []

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
                logger.warning(f"Skipping {ticker} due to missing data")
                continue

            asset_data[ticker] = data
            asset_value = count * purchase_price
            total_weight += asset_value

            asset_return, asset_volatility = self.calculate_asset_metrics(ticker, asset["purchase_date"], start_date, end_date)
            asset_metrics.append((ticker, asset_return, asset_volatility))

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
        logger.info(f"Portfolio values calculated for {len(date_list)} dates")

        daily_returns = np.array([])
        if len(values) > 1:
            daily_returns = np.array([(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))])
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
            data = asset_data.get(ticker)
            if data is not None and not data.empty:
                last_price = data["Close"].iloc[-1]
                if purchase_price > 0:
                    asset_return = (last_price - purchase_price) / purchase_price * 100
                    asset_value = count * purchase_price
                    total_return += asset_return * asset_value
                else:
                    logger.warning(f"Purchase price for {ticker} is zero, skipping in portfolio return calculation")
        portfolio_return = total_return / total_weight if total_weight > 0 else 0.0

        risk_free_rate = 2.0
        sharpe_ratio = (portfolio_return - risk_free_rate) / volatility if volatility > 0 else 0.0

        market_ticker = "^GSPC"
        if progress_dialog:
            progress_dialog.setLabelText("Загрузка данных рынка (S&P 500)...")
            progress_dialog.setValue(len(assets))
            QApplication.processEvents()

        market_data = self.load_historical_prices(market_ticker, start_date, end_date)
        market_returns = np.array([])
        beta = 0.0
        alpha = 0.0
        if market_data is not None and not market_data.empty and len(daily_returns) > 1:
            market_returns = market_data["Close"].pct_change().dropna()
            logger.info(f"Loaded {len(market_returns)} market returns for {market_ticker}")
            market_dates = market_data.index[1:].to_pydatetime()

            portfolio_df = pd.DataFrame({
                "date": date_list[1:],
                "portfolio_returns": daily_returns
            })
            market_df = pd.DataFrame({
                "date": market_dates,
                "market_returns": market_returns.values
            })

            merged_df = pd.merge(portfolio_df, market_df, on="date", how="inner")
            logger.info(f"Merged dataframe has {len(merged_df)} rows")
            if not merged_df.empty:
                portfolio_returns = merged_df["portfolio_returns"].values
                market_returns = merged_df["market_returns"].values
                covariance = np.cov(portfolio_returns, market_returns)[0, 1]
                market_variance = np.var(market_returns)
                beta = covariance / market_variance if market_variance > 0 else 0.0

        expected_return = risk_free_rate + beta * (np.mean(market_returns) * 252 - risk_free_rate) if len(market_returns) > 0 else risk_free_rate
        alpha = portfolio_return - expected_return

        if progress_dialog:
            progress_dialog.setValue(total_steps)

        market_volatility = 15.0 if market_data is None or market_data.empty else np.std(market_returns) * np.sqrt(252)
        market_return = 8.0 if market_data is None or market_data.empty else np.mean(market_returns) * 252

        return date_list, values, [portfolio_return, volatility, alpha, beta, sharpe_ratio, mean_return], asset_metrics, (market_return, market_volatility)

    def calculate_correlation_matrix(self, assets):
        """Расчёт матрицы корреляции между активами."""
        if not assets or len(assets) < 2:
            logger.warning("Not enough assets for correlation matrix")
            return None

        start_date = min(datetime.strptime(asset["purchase_date"], "%Y-%m-%d") for asset in assets)
        end_date = datetime.now()

        data_dict = {}
        for asset in assets:
            ticker = asset["ticker"]
            data = self.load_historical_prices(ticker, start_date, end_date)
            if data is not None and not data.empty:
                daily_returns = data["Close"].pct_change().dropna() * 100
                if not daily_returns.empty:
                    data_dict[ticker] = daily_returns
                else:
                    logger.warning(f"No valid returns data for {ticker}")
            else:
                logger.warning(f"No data for {ticker}")

        if len(data_dict) < 2:
            logger.warning("Not enough valid assets for correlation matrix")
            return None

        df_list = []
        for ticker in data_dict:
            df = pd.DataFrame({
                "date": data_dict[ticker].index.to_pydatetime(),
                ticker: data_dict[ticker].values
            })
            df_list.append(df)

        merged_df = df_list[0]
        for df in df_list[1:]:
            merged_df = pd.merge(merged_df, df, on="date", how="inner")

        if merged_df.empty:
            logger.warning("Merged dataframe is empty, correlation matrix cannot be calculated")
            return None

        returns_data = merged_df.drop(columns=["date"]).values.T
        correlation_matrix = np.corrcoef(returns_data)
        return list(data_dict.keys()), correlation_matrix

    def load_data(self):
        """Загрузка данных из базы и отображение метрик и графиков."""
        assets = self.db_ops.get_assets(self.portfolio_id)
        if not assets:
            logger.warning("No assets found for portfolio")
            return

        progress = QProgressDialog("Загрузка данных...", "Отмена", 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setCancelButton(None)

        dates, values, metrics, asset_metrics, market_metrics = self.calculate_portfolio_metrics(assets, progress)
        progress.close()

        portfolio_return, volatility, alpha, beta, sharpe_ratio, mean_return = metrics
        market_return, market_volatility = market_metrics

        logger.info(f"Portfolio metrics: return={portfolio_return}, volatility={volatility}, alpha={alpha}, beta={beta}, sharpe={sharpe_ratio}, mean_return={mean_return}")

        for metric_name, metric_value in [
            ("portfolio_return", portfolio_return),
            ("volatility", volatility),
            ("alpha", alpha),
            ("beta", beta),
            ("sharpe_ratio", sharpe_ratio),
            ("mean_return", mean_return)
        ]:
            if np.isnan(metric_value) or np.isinf(metric_value):
                logger.warning(f"{metric_name} is NaN or inf, setting to 0: {metric_value}")
                metric_value = 0.0

        for i, (ticker, asset_return, asset_volatility) in enumerate(asset_metrics):
            if np.isnan(asset_return) or np.isinf(asset_return):
                logger.warning(f"Asset return for {ticker} is NaN or inf, setting to 0: {asset_return}")
                asset_return = 0.0
            if np.isnan(asset_volatility) or np.isinf(asset_volatility) or asset_volatility <= 0:
                logger.warning(f"Asset volatility for {ticker} is NaN, inf, or <= 0, setting to 0.01: {asset_volatility}")
                asset_volatility = 0.01
            asset_metrics[i] = (ticker, asset_return, asset_volatility)

        self.ui.metrics.setRowCount(6)
        metric_data = {
            "Доходность (%)": portfolio_return,
            "Волатильность (%)": volatility,
            "Альфа": alpha,
            "Бета": beta,
            "Шарпа": sharpe_ratio,
            "Средняя доходность (%)": mean_return
        }
        row = 0
        for metric, value in metric_data.items():
            item_metric = QTableWidgetItem(metric)
            item_value = QTableWidgetItem(f"{value:.2f}")
            self.ui.metrics.setItem(row, 0, item_metric)
            self.ui.metrics.setItem(row, 1, item_value)
            row += 1

        self.ui.metrics.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ui.metrics.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # График Риск/Доходность (портфель, активы и бенчмарк)
        portfolio_series = QScatterSeries()
        portfolio_volatility = max(volatility, 0.01)
        portfolio_series.append(portfolio_volatility, portfolio_return)
        portfolio_series.setName("Портфель")
        portfolio_series.setMarkerSize(15.0)
        portfolio_series.setColor(QColor("red"))

        assets_series = QScatterSeries()
        valid_volatilities = [portfolio_volatility, market_volatility]
        valid_returns = [portfolio_return, market_return]

        for ticker, asset_return, asset_volatility in asset_metrics:
            assets_series.append(asset_volatility, asset_return)
            assets_series.setMarkerSize(10.0)
            assets_series.setName(f"{ticker}")
            valid_volatilities.append(asset_volatility)
            valid_returns.append(asset_return)

        benchmark_series = QScatterSeries()
        market_volatility = max(market_volatility, 0.01)
        benchmark_series.append(market_volatility, market_return)
        benchmark_series.setName("S&P 500")
        benchmark_series.setMarkerSize(12.0)
        benchmark_series.setColor(QColor("green"))

        chart = QChart()
        chart.addSeries(portfolio_series)
        chart.addSeries(assets_series)
        chart.addSeries(benchmark_series)
        chart.setTitle("Риск/Доходность портфеля, активов и бенчмарка")

        axis_x = QLogValueAxis()
        axis_x.setTitleText("Волатильность (%)")
        axis_x.setLabelFormat("%.1f")
        max_volatility = max(valid_volatilities) * 1.5 if valid_volatilities else 1.0
        min_volatility = min(valid_volatilities) * 0.5 if valid_volatilities else 0.01
        axis_x.setRange(min_volatility, max_volatility)
        chart.addAxis(axis_x, Qt.AlignBottom)
        portfolio_series.attachAxis(axis_x)
        assets_series.attachAxis(axis_x)
        benchmark_series.attachAxis(axis_x)

        axis_y = QValueAxis()
        axis_y.setTitleText("Доходность (%)")
        axis_y.setLabelFormat("%.1f")
        max_return = max(valid_returns + [0]) + 10 if valid_returns else 10
        min_return = min(valid_returns + [-20], default=-20) - 10
        axis_y.setRange(min_return, max_return)
        chart.addAxis(axis_y, Qt.AlignLeft)
        portfolio_series.attachAxis(axis_y)
        assets_series.attachAxis(axis_y)
        benchmark_series.attachAxis(axis_y)

        chart.legend().setVisible(True)
        chart.legend().setAlignment(Qt.AlignBottom)

        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.Antialiasing)
        view_width = self.ui.riskprofit.width()
        view_height = self.ui.riskprofit.height()
        chart_view.setFixedSize(view_width - 10, view_height - 10)
        scene = QGraphicsScene(0, 0, view_width, view_height)
        scene.addWidget(chart_view)
        self.ui.riskprofit.setScene(scene)

        # Матрица корреляции
        tickers, correlation_matrix = self.calculate_correlation_matrix(assets) or ([], np.array([]))
        view_width = self.ui.corelations.width()
        view_height = self.ui.corelations.height()
        scene = QGraphicsScene(0, 0, view_width, view_height)

        if tickers and correlation_matrix.size > 0:
            n = len(tickers)
            cell_size = min(view_width / (n + 1), view_height / (n + 1))
            offset_x = cell_size
            offset_y = cell_size

            for i in range(n):
                label_x = QGraphicsTextItem(tickers[i])
                label_x.setPos(offset_x + i * cell_size + cell_size / 4, 0)
                scene.addItem(label_x)
                label_y = QGraphicsTextItem(tickers[i])
                label_y.setPos(0, offset_y + i * cell_size + cell_size / 4)
                scene.addItem(label_y)

            for i in range(n):
                for j in range(n):
                    value = correlation_matrix[i][j]
                    hue = int(240 * (1 - value) / 2)
                    color = QColor.fromHsv(hue, 255, 255)
                    rect = QGraphicsRectItem(offset_x + j * cell_size, offset_y + i * cell_size, cell_size, cell_size)
                    rect.setBrush(color)
                    rect.setPen(QPen(Qt.black))
                    scene.addItem(rect)

                    text = QGraphicsTextItem(f"{value:.2f}")
                    text.setDefaultTextColor(Qt.white if hue < 120 else Qt.black)
                    text_width = text.boundingRect().width()
                    text_height = text.boundingRect().height()
                    text.setPos(offset_x + j * cell_size + (cell_size - text_width) / 2, offset_y + i * cell_size + (cell_size - text_height) / 2)
                    scene.addItem(text)

        else:
            text = scene.addText("Нет данных или недостаточно активов")
            text.setPos(view_width / 2 - text.boundingRect().width() / 2, view_height / 2 - text.boundingRect().height() / 2)

        self.ui.corelations.setScene(scene)

        # График Историческая динамика
        if dates and values:
            series = QLineSeries()
            for i, (date, value) in enumerate(zip(dates, values)):
                qdate = QDateTime(date)
                series.append(qdate.toMSecsSinceEpoch(), value)

            chart = QChart()
            chart.addSeries(series)
            chart.setTitle("Историческая динамика стоимости")

            axis_x = QDateTimeAxis()
            axis_x.setFormat("dd-MM-yyyy")
            axis_x.setTitleText("Дата")
            chart.addAxis(axis_x, Qt.AlignBottom)
            series.attachAxis(axis_x)

            axis_y = QValueAxis()
            axis_y.setLabelFormat("%.2f")
            axis_y.setTitleText("Стоимость ($)")
            chart.addAxis(axis_y, Qt.AlignLeft)
            series.attachAxis(axis_y)

            chart_view = QChartView(chart)
            chart_view.setRenderHint(QPainter.Antialiasing)
            view_width = self.ui.historicalpriceportfoliochart.width()
            view_height = self.ui.historicalpriceportfoliochart.height()
            chart_view.setFixedSize(view_width - 10, view_height - 10)
            scene = QGraphicsScene(0, 0, view_width, view_height)
            scene.addWidget(chart_view)
            self.ui.historicalpriceportfoliochart.setScene(scene)
        else:
            view_width = self.ui.historicalpriceportfoliochart.width()
            view_height = self.ui.historicalpriceportfoliochart.height()
            scene = QGraphicsScene(0, 0, view_width, view_height)
            text = scene.addText("Нет данных для отображения")
            text.setPos(view_width / 2 - text.boundingRect().width() / 2, view_height / 2 - text.boundingRect().height() / 2)
            self.ui.historicalpriceportfoliochart.setScene(scene)

    def closeEvent(self, event):
        self.db_ops.close()
        event.accept()