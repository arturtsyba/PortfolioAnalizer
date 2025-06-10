from PyQt5.QtWidgets import QMainWindow, QMessageBox
from PyQt5.QtCore import QDate
from windows.ui_add_assets import Ui_AssetsWindow
from data.db_operations import DBOperations
from datetime import date

class AddAssetsWindow(QMainWindow):
    def __init__(self, portfolio_id=None):
        super().__init__()
        self.ui = Ui_AssetsWindow()
        self.ui.setupUi(self)
        self.db_ops = DBOperations()
        self.portfolio_id = portfolio_id  # Получаем portfolio_id из главного окна
        self.connect_signals()

    def connect_signals(self):
        """Подключение сигналов к кнопкам."""
        self.ui.pushButton_2.clicked.connect(self.add_asset)
        self.ui.pushButton.clicked.connect(self.clear_fields)

    def add_asset(self):
        """Сохранение актива в базу данных."""
        if not self.portfolio_id:
            QMessageBox.critical(self, "Ошибка", "Не выбран портфель")
            return

        ticker = self.ui.textEdit.toPlainText().strip()
        asset_type = self.ui.comboBox.currentText()
        count_assets = self.ui.textEdit_2.toPlainText().strip()

        # Автоматически добавляем суффикс -USD для криптовалют
        if asset_type == "Криптовалюты" and not ticker.endswith("-USD"):
            ticker = f"{ticker}-USD"
        # Автоматически добавляем суффикс -ru для российских акций
        elif asset_type == "Акции" and not ticker.endswith("-ru"):
            ticker = f"{ticker}-ru"

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

        success = self.db_ops.add_asset(self.portfolio_id, ticker, asset_type, count_assets, price, purchase_date)
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
        self.db_ops.close()
        event.accept()