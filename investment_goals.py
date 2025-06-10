from PyQt5.QtWidgets import QMainWindow, QMessageBox
from windows.ui_investment_goals import Ui_InvestmentGoals
from data.db_operations import DBOperations

class InvestmentGoalsWindow(QMainWindow):
    def __init__(self, user_id):
        super().__init__()
        self.ui = Ui_InvestmentGoals()
        self.ui.setupUi(self)
        self.user_id = user_id
        self.db_ops = DBOperations()
        self.connect_signals()
        self.load_existing_data()

    def connect_signals(self):
        """Подключение сигналов к кнопкам."""
        self.ui.saveButton.clicked.connect(self.save_data)

    def load_existing_data(self):
        """Загрузка существующих данных пользователя."""
        user_data = self.db_ops.get_user_by_id(self.user_id)
        print(f"Loaded user data: {user_data}")
        if user_data:
            horizon = user_data["horizon"]
            goal = user_data["investment_goal"]
            print(f"Setting horizon: {horizon}")
            self.ui.horizonLineEdit.setText(str(horizon) if horizon is not None else "")
            
            print(f"Setting goal: {goal}")
            print(f"Available goals in combo box: {[self.ui.goalComboBox.itemText(i) for i in range(self.ui.goalComboBox.count())]}")
            if goal:
                index = self.ui.goalComboBox.findText(goal)
                print(f"Found goal index: {index}")
                if index >= 0:
                    self.ui.goalComboBox.setCurrentIndex(index)
                else:
                    print(f"Goal '{goal}' not found in combo box")
            else:
                self.ui.goalComboBox.setCurrentIndex(0)  # Выбираем первый элемент, если цель не задана

    def save_data(self):
        """Сохранение данных в базу."""
        # Проверка горизонта
        horizon_text = self.ui.horizonLineEdit.text().strip()
        if not horizon_text:
            QMessageBox.critical(self, "Ошибка", "Введите инвестиционный горизонт")
            return
        try:
            horizon = int(horizon_text)
            if horizon <= 0:
                raise ValueError("Инвестиционный горизонт должен быть положительным числом")
        except ValueError as e:
            QMessageBox.critical(self, "Ошибка", str(e))
            return

        # Проверка цели
        goal = self.ui.goalComboBox.currentText()
        if not goal:
            QMessageBox.critical(self, "Ошибка", "Выберите цель инвестирования")
            return

        # Сохранение данных
        print(f"Saving data: user_id={self.user_id}, horizon={horizon}, goal={goal}")
        success = self.db_ops.update_user(self.user_id, investment_goal=goal, horizon=horizon)
        if success:
            QMessageBox.information(self, "Успех", "Данные успешно сохранены")
            self.close()
        else:
            QMessageBox.critical(self, "Ошибка", "Не удалось сохранить данные")

    def closeEvent(self, event):
        self.db_ops.close()
        event.accept()