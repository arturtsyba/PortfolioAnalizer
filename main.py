#Убираем предупреждение TF в консоли
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableView, QMenu, QComboBox, QInputDialog, QMessageBox
from PyQt5.QtCore import Qt, QAbstractTableModel
from PyQt5.QtGui import QStandardItem, QStandardItemModel
from PyQt5 import QtWidgets
from PyQt5.QtChart import QChart, QChartView, QPieSeries
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsTextItem
from PyQt5.QtGui import QPainter, QFont

from windows.ui_main_window import Ui_MainWindow
from risk_window_controller import RiskWindowController
from addassets import AddAssetsWindow
from editassets import EditAssetsWindow
from investment_goals import InvestmentGoalsWindow
from indicators import IndicatorsWindow
from data.db_operations import DBOperations
import yfinance as yf
import apimoex
import requests
from datetime import datetime
from optimization import OptimizationWindowLogic
from profit_logic import ProfitbleWindowLogic

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.db_ops = DBOperations()
        self.add_window = None
        self.edit_window = None
        self.goals_window = None
        self.indicators_window = None
        self.risk_window = None
        self.optimization_window = None
        self.profitble_window = None
        self.current_portfolio_id = 1
        self.user_id = 1

        # Подключение сигналов
        self.ui.NewButton.clicked.connect(self.create_new_portfolio)
        self.ui.delButton.clicked.connect(self.delete_current_portfolio)
        self.ui.AddButton.clicked.connect(self.open_add_window)
        self.ui.dataforNNbutton.clicked.connect(self.open_goals_window)
        self.ui.pushButton_2.clicked.connect(self.open_indicators_window)
        self.ui.pushButton_3.clicked.connect(self.open_risk_window)
        self.ui.pushButton_4.clicked.connect(self.open_optimization_window)
        self.ui.pushButton_5.clicked.connect(self.open_profitble_window)
        self.ui.SaveButton_3.clicked.connect(self.load_assets)
        self.ui.comboBox.currentIndexChanged.connect(self.on_portfolio_changed)

        # Настройка модели и таблицы
        self.model = QStandardItemModel(self)
        self.ui.tableView.setModel(self.model)
        self.ui.tableView.setContextMenuPolicy(Qt.CustomContextMenu)
        self.ui.tableView.customContextMenuRequested.connect(self.show_context_menu)

        # Загрузка портфелей
        self.load_portfolios()
        self.load_assets()

    def open_profitble_window(self):
        if self.current_portfolio_id:
            self.profitble_window = ProfitbleWindowLogic(
                db_ops=self.db_ops,
                user_id=self.user_id,
                current_portfolio_id=self.current_portfolio_id
            )
            self.profitble_window.show()
        else:
            QMessageBox.warning(self, "Предупреждение", "Сначала выберите портфель")

    def open_optimization_window(self):
        if self.current_portfolio_id:
            self.optimization_window = OptimizationWindowLogic(
                db_ops=self.db_ops,
                user_id=self.user_id,
                current_portfolio_id=self.current_portfolio_id
            )
            self.optimization_window.show()
        else:
            QMessageBox.warning(self, "Предупреждение", "Сначала выберите портфель")

    def open_risk_window(self):
        if self.current_portfolio_id:
            self.risk_window = RiskWindowController(self.db_ops, self.user_id, self.current_portfolio_id)
            self.risk_window.show()
        else:
            QMessageBox.warning(self, "Предупреждение", "Сначала выберите портфель")

    def create_new_portfolio(self):
        portfolio_name, ok = QInputDialog.getText(self, "Создать портфель", "Введите название портфеля:")
        if ok and portfolio_name:
            user_id = 1
            success = self.db_ops.add_portfolio(user_id, portfolio_name)
            if success:
                QMessageBox.information(self, "Успех", "Портфель успешно создан")
                self.load_portfolios()
            else:
                QMessageBox.critical(self, "Ошибка", "Не удалось создать портфель")

    def delete_current_portfolio(self):
        if not self.current_portfolio_id:
            QMessageBox.warning(self, "Предупреждение", "Нет портфеля для удаления")
            return

        reply = QMessageBox.question(
            self, "Подтверждение", "Вы уверены, что хотите удалить этот портфель? Все связанные активы будут удалены.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.No:
            return

        success = self.db_ops.delete_portfolio(self.current_portfolio_id)
        if success:
            QMessageBox.information(self, "Успех", "Портфель успешно удален")
            self.load_portfolios()
            self.load_assets()
        else:
            QMessageBox.critical(self, "Ошибка", "Не удалось удалить портфель")

    def open_add_window(self):
        if self.current_portfolio_id:
            self.add_window = AddAssetsWindow(portfolio_id=self.current_portfolio_id)
            self.add_window.show()

    def open_edit_window(self, asset_data):
        self.edit_window = EditAssetsWindow(asset_data=asset_data)
        self.edit_window.show()

    def open_goals_window(self):
        print(f"Opening goals window for user_id: {self.user_id}")
        self.goals_window = InvestmentGoalsWindow(user_id=self.user_id)
        self.goals_window.show()

    def open_indicators_window(self):
        if self.current_portfolio_id:
            print(f"Opening indicators window for user_id: {self.user_id}, portfolio_id: {self.current_portfolio_id}")
            self.indicators_window = IndicatorsWindow(user_id=self.user_id, portfolio_id=self.current_portfolio_id)
            self.indicators_window.show()

    def load_portfolios(self):
        portfolios = self.db_ops.get_portfolios(user_id=1)
        self.ui.comboBox.clear()
        if not portfolios:
            self.ui.comboBox.addItem("Нет портфелей", 0)
            self.current_portfolio_id = None
        else:
            for portfolio in portfolios:
                self.ui.comboBox.addItem(portfolio["portfolio_name"], portfolio["portfolio_id"])
            self.current_portfolio_id = self.ui.comboBox.itemData(0)

    def load_assets(self):
        if not self.current_portfolio_id:
            self.model.clear()
            self.model.setHorizontalHeaderLabels(["Тикер", "Тип", "Кол-во", "Цена покупки", "Дата покупки", "Текущая цена", "Стоимость", "%"])
            self.update_pie_chart()
            self.update_metrics()
            return

        self.model.clear()
        self.model.setHorizontalHeaderLabels(["Тикер", "Тип", "Кол-во", "Цена покупки", "Дата покупки", "Текущая цена", "Стоимость", "%"])
        assets = self.db_ops.get_assets(portfolio_id=self.current_portfolio_id)
        
        for asset in assets:
            ticker = asset["ticker"]
            purchase_price = float(asset["purchase_price"])
            count = float(asset["count_assets"])
            current_price = self.get_current_price(ticker, assets)
            total_value = count * current_price
            change_percent = (current_price - purchase_price) / purchase_price * 100 if purchase_price != 0 else 0.0

            row = [
                QStandardItem(ticker),
                QStandardItem(asset["asset_type"]),
                QStandardItem(str(asset["count_assets"])),
                QStandardItem(f"{purchase_price:.2f}"),
                QStandardItem(str(asset["purchase_date"])),
                QStandardItem(f"{current_price:.2f}"),
                QStandardItem(f"{total_value:.2f}"),
                QStandardItem(f"{change_percent:.2f}%")
            ]
            self.model.appendRow(row)

        self.ui.tableView.resizeColumnsToContents()
        header = self.ui.tableView.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(QtWidgets.QHeaderView.Stretch)

        self.update_pie_chart()
        self.update_metrics()

    def get_current_price(self, ticker, assets):
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
                        print(f"No current price data for {ticker} on MOEX board {board}, trying next board or purchase price")
            except Exception as e:
                print(f"Error fetching current price for {ticker} on MOEX: {e}, using purchase price")
        else:
            try:
                stock = yf.Ticker(ticker)
                data = stock.history(period="1d")
                if not data.empty:
                    return data["Close"].iloc[-1]
                print(f"No current price data for {ticker} on Yahoo Finance, using purchase price")
            except Exception as e:
                print(f"Error fetching current price for {ticker} on Yahoo Finance: {e}, using purchase price")

        for asset in assets:
            if asset["ticker"] == ticker:
                return float(asset["purchase_price"])
        return 0.0

    def update_pie_chart(self):
        if not self.current_portfolio_id:
            self.ui.graphicsView.setScene(QGraphicsScene())
            return

        assets = self.db_ops.get_assets(portfolio_id=self.current_portfolio_id)
        series = QPieSeries()

        total_value = 0
        current_prices = {}
        for asset in assets:
            ticker = asset["ticker"]
            count = float(asset["count_assets"])
            current_price = self.get_current_price(ticker, assets)
            current_prices[ticker] = current_price
            value = count * current_price
            total_value += value
            series.append(f"{ticker} (${round(value, 2)})", value)

        chart = QChart()
        chart.addSeries(series)
        chart.setTitle("Распределение активов (текущие цены)")
        chart.legend().setAlignment(Qt.AlignBottom)

        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.Antialiasing)

        view_width = self.ui.graphicsView.width()
        view_height = self.ui.graphicsView.height()
        chart_view.setFixedSize(view_width - 10, view_height - 40)

        scene = QGraphicsScene(0, 0, view_width, view_height)
        scene.addWidget(chart_view)

        label_text = f"Общая стоимость портфеля: ${round(total_value, 2)}"
        label = QGraphicsTextItem(label_text)
        label.setFont(QFont("Arial", 9))
        
        label_width = label.boundingRect().width()
        label_height = label.boundingRect().height()
        
        label.setPos(10, view_height - label_height - 10)
        
        if label_width > view_width - 20:
            label.setTextWidth(view_width - 20)
            label_height = label.boundingRect().height()
            label.setPos(10, view_height - label_height - 10)

        scene.setSceneRect(0, 0, max(view_width, label_width + 20), view_height)
        scene.addItem(label)

        self.ui.graphicsView.setScene(scene)

    def update_metrics(self):
        GREEN = "#008000"
        RED = "#8b0000"

        if not self.current_portfolio_id:
            self.ui.coast_basis_label.setText("Пусто")
            self.ui.profit_label.setText("Пусто")
            self.ui.netprofit_label.setText("Пусто")
            self.ui.best_asset.setText("Пусто")
            self.ui.worst_asset.setText("Пусто")
            self.ui.profit_label.setStyleSheet("")
            self.ui.netprofit_label.setStyleSheet("")
            self.ui.best_asset.setStyleSheet("")
            self.ui.worst_asset.setStyleSheet("")
            return

        assets = self.db_ops.get_assets(portfolio_id=self.current_portfolio_id)
        if not assets:
            self.ui.coast_basis_label.setText("Пусто")
            self.ui.profit_label.setText("Пусто")
            self.ui.netprofit_label.setText("Пусто")
            self.ui.best_asset.setText("Пусто")
            self.ui.worst_asset.setText("Пусто")
            self.ui.profit_label.setStyleSheet("")
            self.ui.netprofit_label.setStyleSheet("")
            self.ui.best_asset.setStyleSheet("")
            self.ui.worst_asset.setStyleSheet("")
            return

        cost_basis = 0.0
        current_value = 0.0
        asset_changes = []

        for asset in assets:
            ticker = asset["ticker"]
            count = float(asset["count_assets"])
            purchase_price = float(asset["purchase_price"])
            current_price = self.get_current_price(ticker, assets)
            cost_basis += count * purchase_price
            current_value += count * current_price
            if purchase_price != 0:
                change_percent = (current_price - purchase_price) / purchase_price * 100
            else:
                change_percent = 0.0
            asset_changes.append((ticker, change_percent))

        profit = current_value - cost_basis
        if cost_basis != 0:
            net_profit = (profit / cost_basis) * 100
        else:
            net_profit = 0.0

        if asset_changes:
            asset_changes.sort(key=lambda x: x[1], reverse=True)
            best_asset = asset_changes[0][0]
            best_asset_change = asset_changes[0][1]
            worst_asset = asset_changes[-1][0]
            worst_asset_change = asset_changes[-1][1]
        else:
            best_asset = "Пусто"
            worst_asset = "Пусто"
            best_asset_change = 0.0
            worst_asset_change = 0.0

        self.ui.coast_basis_label.setText(f"${cost_basis:.2f}")
        self.ui.profit_label.setText(f"${profit:.2f}")
        if profit > 0:
            self.ui.profit_label.setStyleSheet(f"color: {GREEN};")
        else:
            self.ui.profit_label.setStyleSheet(f"color: {RED};")

        self.ui.netprofit_label.setText(f"{net_profit:.2f}%")
        if net_profit > 0:
            self.ui.netprofit_label.setStyleSheet(f"color: {GREEN};")
        else:
            self.ui.netprofit_label.setStyleSheet(f"color: {RED};")

        self.ui.best_asset.setText(best_asset)
        if best_asset_change > 0:
            self.ui.best_asset.setStyleSheet(f"color: {GREEN};")
        else:
            self.ui.best_asset.setStyleSheet(f"color: {RED};")

        self.ui.worst_asset.setText(worst_asset)
        if worst_asset_change > 0:
            self.ui.worst_asset.setStyleSheet(f"color: {GREEN};")
        else:
            self.ui.worst_asset.setStyleSheet(f"color: {RED};")

    def on_portfolio_changed(self, index):
        portfolio_id = self.ui.comboBox.itemData(index)
        if portfolio_id == 0:
            self.current_portfolio_id = None
        else:
            self.current_portfolio_id = portfolio_id
        self.load_assets()

    def show_context_menu(self, pos):
        if not self.current_portfolio_id:
            return

        menu = QMenu()
        edit_action = menu.addAction("Редактировать")
        delete_action = menu.addAction("Удалить")
        action = menu.exec_(self.ui.tableView.viewport().mapToGlobal(pos))
        
        selected = self.ui.tableView.selectedIndexes()
        if not selected:
            return
        
        row = selected[0].row()
        assets = self.db_ops.get_assets(portfolio_id=self.current_portfolio_id)
        asset_data = assets[row] if row < len(assets) else None
        
        if not asset_data:
            QMessageBox.warning(self, "Предупреждение", "Выбранный актив не найден")
            return
        
        if action == edit_action:
            self.open_edit_window(asset_data)
        elif action == delete_action:
            self.delete_asset(asset_data["asset_id"])

    def delete_asset(self, asset_id):
        success = self.db_ops.delete_asset(asset_id)
        if success:
            QMessageBox.information(self, "Успех", "Актив успешно удален")
            self.load_assets()
        else:
            QMessageBox.critical(self, "Ошибка", "Не удалось удалить актив")

    def closeEvent(self, event):
        if hasattr(self, 'db_ops'):
            self.db_ops.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())

#Строк кода примерно 3595