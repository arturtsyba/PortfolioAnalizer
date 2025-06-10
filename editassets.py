from PyQt5.QtWidgets import QMainWindow, QMessageBox
from PyQt5.QtCore import QDate
from windows.ui_add_assets import Ui_AssetsWindow
from data.db_operations import DBOperations
from datetime import date

class EditAssetsWindow(QMainWindow):
    def __init__(self, asset_data=None):
        super().__init__()
        self.ui = Ui_AssetsWindow()
        self.ui.setupUi(self)
        self.db_ops = DBOperations()
        self.asset_data = asset_data  # Данные актива для редактирования
        self.connect_signals()

        # Если данные актива переданы, загружаем их для редактирования
        if self.asset_data:
            self.load_asset_data()
            self.setWindowTitle("Редактирование актива")
        else:
            self.setWindowTitle("Добавление актива")

    def connect_signals(self):
        """Подключение сигналов к кнопкам."""
        self.ui.pushButton_2.clicked.connect(self.save_asset)  # Кнопка "Сохранить"
        self.ui.pushButton.clicked.connect(self.clear_fields)  # Кнопка "Очистить"

    def load_asset_data(self):
        """Загрузка данных существующего актива для редактирования."""
        if self.asset_data:
            self.ui.textEdit.setPlainText(self.asset_data["ticker"])
            self.ui.comboBox.setCurrentText(self.asset_data["asset_type"])
            self.ui.textEdit_2.setPlainText(str(self.asset_data["count_assets"]))
            self.ui.textEdit_3.setPlainText(str(self.asset_data["purchase_price"]))
            self.ui.dateEdit.setDate(QDate.fromString(self.asset_data["purchase_date"], "yyyy-MM-dd"))

    def save_asset(self):
        """Сохранение или обновление актива в базе данных."""
        ticker = self.ui.textEdit.toPlainText().strip()
        asset_type = self.ui.comboBox.currentText()
        count_assets = self.ui.textEdit_2.toPlainText().strip()
        try:
            price = float(self.ui.textEdit_3.toPlainText().strip())
            if price <= 0:
                raise ValueError("Цена должна быть положительной")
        except ValueError as e:
            QMessageBox.critical(self, "Ошибка", str(e))
            return

        purchase_date = self.ui.dateEdit.date().toPyDate()
        if purchase_date > date.today():
            QMessageBox.critical(self, "Ошибка", "Дата покупки не может быть в будущем")
            return

        portfolio_id = 1  # Фиксированный ID портфеля

        if self.asset_data:
            # Обновление существующего актива
            success = self.db_ops.update_asset(self.asset_data["asset_id"], ticker, asset_type, count_assets, price, purchase_date)
            if success:
                QMessageBox.information(self, "Успех", "Актив успешно обновлен")
                self.close()
            else:
                QMessageBox.critical(self, "Ошибка", "Не удалось обновить актив")
        else:
            # Добавление нового актива
            success = self.db_ops.add_asset(portfolio_id, ticker, asset_type, count_assets, price, purchase_date)
            if success:
                QMessageBox.information(self, "Успех", "Актив успешно добавлен")
                self.clear_fields()
            else:
                QMessageBox.critical(self, "Ошибка", "Не удалось добавить актив")

    def clear_fields(self):
        """Сброс всех полей ввода."""
        self.ui.textEdit.clear()
        self.ui.comboBox.setCurrentIndex(0)
        self.ui.textEdit_2.clear()
        self.ui.textEdit_3.clear()
        self.ui.dateEdit.setDate(QDate.currentDate())

    def closeEvent(self, event):
        """Закрытие соединения с базой при закрытии окна."""
        self.db_ops.close()
        event.accept()