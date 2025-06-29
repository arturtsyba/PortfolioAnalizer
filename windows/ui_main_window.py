from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("Система анализа инвестиционного портфеля на основе неронных сетей")
        MainWindow.resize(1087, 750)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.labelmain = QtWidgets.QLabel(self.centralwidget)
        self.labelmain.setGeometry(QtCore.QRect(150, -20, 811, 71))
        font = QtGui.QFont()
        font.setFamily("MS Sans Serif")
        font.setPointSize(22)
        font.setBold(True)
        font.setWeight(75)
        self.labelmain.setFont(font)
        self.labelmain.setStyleSheet("color: #611BF8;\n"
"")
        self.labelmain.setObjectName("labelmain")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(310, 20, 531, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(60, 60, 261, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(340, 60, 231, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(590, 60, 211, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton_4.setFont(font)
        self.pushButton_4.setObjectName("pushButton_4")
        self.tableView = QtWidgets.QTableView(self.centralwidget)
        self.tableView.setGeometry(QtCore.QRect(10, 260, 711, 361))
        self.tableView.setObjectName("tableView")
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(730, 220, 341, 451))
        self.graphicsView.setObjectName("graphicsView")
        self.AddButton = QtWidgets.QPushButton(self.centralwidget)
        self.AddButton.setGeometry(QtCore.QRect(260, 630, 191, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.AddButton.setFont(font)
        self.AddButton.setObjectName("AddButton")
        self.NewButton = QtWidgets.QPushButton(self.centralwidget)
        self.NewButton.setGeometry(QtCore.QRect(210, 130, 151, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.NewButton.setFont(font)
        self.NewButton.setObjectName("NewButton")
        self.dataforNNbutton = QtWidgets.QPushButton(self.centralwidget)
        self.dataforNNbutton.setGeometry(QtCore.QRect(800, 130, 221, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.dataforNNbutton.setFont(font)
        self.dataforNNbutton.setObjectName("dataforNNbutton")
        self.SaveButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.SaveButton_3.setGeometry(QtCore.QRect(630, 220, 91, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.SaveButton_3.setFont(font)
        self.SaveButton_3.setObjectName("SaveButton_3")
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(10, 130, 191, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.comboBox.setFont(font)
        self.comboBox.setObjectName("comboBox")
        self.delButton = QtWidgets.QPushButton(self.centralwidget)
        self.delButton.setGeometry(QtCore.QRect(370, 130, 161, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.delButton.setFont(font)
        self.delButton.setObjectName("delButton")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(300, 180, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.profit_label = QtWidgets.QLabel(self.centralwidget)
        self.profit_label.setGeometry(QtCore.QRect(400, 180, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.profit_label.setFont(font)
        self.profit_label.setObjectName("profit_label")
        self.netprofit_label = QtWidgets.QLabel(self.centralwidget)
        self.netprofit_label.setGeometry(QtCore.QRect(670, 180, 131, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.netprofit_label.setFont(font)
        self.netprofit_label.setObjectName("netprofit_label")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(500, 180, 171, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(10, 220, 181, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.best_asset = QtWidgets.QLabel(self.centralwidget)
        self.best_asset.setGeometry(QtCore.QRect(190, 220, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.best_asset.setFont(font)
        self.best_asset.setObjectName("best_asset")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(300, 220, 151, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.worst_asset = QtWidgets.QLabel(self.centralwidget)
        self.worst_asset.setGeometry(QtCore.QRect(450, 220, 121, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.worst_asset.setFont(font)
        self.worst_asset.setObjectName("worst_asset")
        self.coast_basis_label = QtWidgets.QLabel(self.centralwidget)
        self.coast_basis_label.setGeometry(QtCore.QRect(190, 180, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.coast_basis_label.setFont(font)
        self.coast_basis_label.setObjectName("coast_basis_label")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(10, 180, 181, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(820, 60, 211, 41))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.pushButton_5.setFont(font)
        self.pushButton_5.setObjectName("pushButton_5")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(50, 670, 941, 41))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1087, 21))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        self.menu_2 = QtWidgets.QMenu(self.menubar)
        self.menu_2.setObjectName("menu_2")
        self.menu_3 = QtWidgets.QMenu(self.menubar)
        self.menu_3.setObjectName("menu_3")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.action = QtWidgets.QAction(MainWindow)
        self.action.setObjectName("action")
        self.action_2 = QtWidgets.QAction(MainWindow)
        self.action_2.setObjectName("action_2")
        self.action_3 = QtWidgets.QAction(MainWindow)
        self.action_3.setObjectName("action_3")
        self.action_4 = QtWidgets.QAction(MainWindow)
        self.action_4.setObjectName("action_4")
        self.menu.addAction(self.action)
        self.menu.addAction(self.action_2)
        self.menu.addAction(self.action_3)
        self.menu.addAction(self.action_4)
        self.menubar.addAction(self.menu.menuAction())
        self.menubar.addAction(self.menu_3.menuAction())
        self.menubar.addAction(self.menu_2.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Анализ инвестиционного портфеля"))
        self.labelmain.setText(_translate("MainWindow", "Интеллектуальный анализ инвестиционного портфеля"))
        self.label.setText(_translate("MainWindow", "Продвинутый аналитический инструмент на основе нейронных сетей"))
        self.pushButton_2.setText(_translate("MainWindow", "Анализ и метрики"))
        self.pushButton_3.setText(_translate("MainWindow", "Определение риска портфеля"))
        self.pushButton_4.setText(_translate("MainWindow", "Оптимизация портфеля"))
        self.AddButton.setText(_translate("MainWindow", "Добавить актив"))
        self.NewButton.setText(_translate("MainWindow", "Создать портфель"))
        self.dataforNNbutton.setText(_translate("MainWindow", "Доп. данные"))
        self.SaveButton_3.setText(_translate("MainWindow", "Обновить"))
        self.delButton.setText(_translate("MainWindow", "Удалить портфель"))
        self.label_2.setText(_translate("MainWindow", "Прибыль:"))
        self.profit_label.setText(_translate("MainWindow", "Пусто"))
        self.netprofit_label.setText(_translate("MainWindow", "Пусто"))
        self.label_3.setText(_translate("MainWindow", "Чистая прибыль:"))
        self.label_4.setText(_translate("MainWindow", "Самый доходный:"))
        self.best_asset.setText(_translate("MainWindow", "Пусто"))
        self.label_5.setText(_translate("MainWindow", "Самый худший:"))
        self.worst_asset.setText(_translate("MainWindow", "Пусто"))
        self.coast_basis_label.setText(_translate("MainWindow", "Пусто"))
        self.label_6.setText(_translate("MainWindow", "Базовая ценность:"))
        self.pushButton_5.setText(_translate("MainWindow", "Прогнозирование доходности"))
        self.label_7.setText(_translate("MainWindow", "Внимание! Это не инвестиционная рекомендация. Прогнозы нейросети носят ознакомительный характер и не гарантируют результат."))
        self.menu.setTitle(_translate("MainWindow", "Анализ "))
        self.menu_2.setTitle(_translate("MainWindow", "О программе"))
        self.menu_3.setTitle(_translate("MainWindow", "Настройки"))
        self.action.setText(_translate("MainWindow", "Метрики портфеля"))
        self.action_2.setText(_translate("MainWindow", "Рекомендации по отдельным активам"))
        self.action_3.setText(_translate("MainWindow", "Оптимизация портфеля"))
        self.action_4.setText(_translate("MainWindow", "Прогноз доходности"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
