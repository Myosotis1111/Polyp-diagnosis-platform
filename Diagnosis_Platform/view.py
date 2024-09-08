from threading import Thread
from PySide2.QtCore import QSize
from PySide2.QtGui import QPixmap, Qt, QColor, QBrush
from PySide2.QtUiTools import QUiLoader
from PySide2.QtWidgets import QTableWidgetItem, QHeaderView
from model import Model


class MainView:

    def __init__(self):

        self.m = Model()
        self.ms = self.m.ms
        self.ui = QUiLoader().load('demo.ui')
        super().__init__()

        self.ms.update_image.connect(self.update_image)
        self.ms.update_text.connect(self.update_text)
        self.ms.update_tab_status.connect(self.update_tab_status)
        self.ms.update_feature_table.connect(self.update_feature_table)
        self.ms.add_importance.connect(self.add_importance)
        self.ms.set_cell_color.connect(self.set_cell_color)
        self.ms.update_processBar.connect(self.update_progressBar)

        self.ui.uploadButton.clicked.connect(self.on_uploadButton_clicked)
        self.ui.segmentButton.clicked.connect(self.on_segmentButton_clicked)
        self.ui.evaluateButton.clicked.connect(self.on_evaluateButton_clicked)
        self.ui.classifyButton.clicked.connect(self.on_classifyButton_clicked)
        self.ui.exportButton.clicked.connect(self.on_exportButton_clicked)
        self.ui.batchButton.clicked.connect(self.on_batchButton_clicked)

        self.ui.tableWidget.setColumnCount(3)
        self.ui.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # Set fixed row height
        row_height = 15  # Adjust this value as needed
        self.ui.tableWidget.verticalHeader().setDefaultSectionSize(row_height)

        # Make header font bold
        font = self.ui.tableWidget.horizontalHeader().font()
        font.setBold(True)
        self.ui.tableWidget.horizontalHeader().setFont(font)

    def on_uploadButton_clicked(self):

        # Initialize charts and text
        empty_image = QPixmap(QSize(0, 0))
        empty_image.fill(Qt.transparent)
        self.ui.currentView.setPixmap(empty_image)
        self.ui.inputView.setPixmap(empty_image)
        self.ui.maskView.setPixmap(empty_image)
        self.ui.maskedView.setPixmap(empty_image)
        self.ui.evalView.setPixmap(empty_image)
        self.ui.enhancedView.setPixmap(empty_image)

        self.ui.resultBrowser.setText("")
        self.ui.evalBrowser.setText("")

        tab_bar = self.ui.tabWidget.tabBar()
        for i in range(tab_bar.count()):
            tab_bar.setTabTextColor(i, QColor('black'))

        row_count = self.ui.tableWidget.rowCount()
        if row_count > 0:
            self.ui.tableWidget.setRowCount(0)

        self.m.upload()

    def on_segmentButton_clicked(self):

        self.m.segment()

    def on_evaluateButton_clicked(self):

        self.m.evaluate()

    def on_classifyButton_clicked(self):

        row_count = self.ui.tableWidget.rowCount()
        if row_count > 0:
            self.ui.tableWidget.setRowCount(0)

        self.m.extract_feature()

        self.m.classify()

        self.ui.tableWidget.sortItems(2, Qt.DescendingOrder)

        self.ui.tableWidget.resizeColumnsToContents()

    def on_exportButton_clicked(self):

        self.m.export()

    def on_batchButton_clicked(self):

        def batchProcessingThread():
            self.ui.progressBar.reset()
            self.m.batch_process(self.m.batch_input_dir, self.m.batch_output_dir)

        t = Thread(target=batchProcessingThread)
        t.start()

    def update_image(self, pixmap, name):

        # Display images in GUI
        if name == "input":
            self.ui.inputView.setPixmap(pixmap)
            self.ui.inputView.setScaledContents(True)
        elif name == "mask":
            self.ui.maskView.setPixmap(pixmap)
            self.ui.maskView.setScaledContents(True)
        elif name == "eval":
            self.ui.evalView.setPixmap(pixmap)
            self.ui.evalView.setScaledContents(True)
        elif name == "enhanced":
            self.ui.enhancedView.setPixmap(pixmap)
            self.ui.enhancedView.setScaledContents(True)
        elif name == "masked":
            self.ui.maskedView.setPixmap(pixmap)
            self.ui.maskedView.setScaledContents(True)
        elif name == "current":
            self.ui.currentView.setPixmap(pixmap)
            self.ui.currentView.setScaledContents(True)

    def update_text(self, text, name, color="none"):

        def append_text(browser, text):
            current_text = browser.toPlainText()
            new_text = current_text + "\n" + text if current_text else text
            browser.setText(new_text)

            # Move the cursor to the end to scroll to the bottom
            cursor = browser.textCursor()
            cursor.movePosition(cursor.End)
            browser.setTextCursor(cursor)
            browser.ensureCursorVisible()

        if name == "clsResult":
            self.ui.resultBrowser.setText(text)
            if color == "green":
                formatted_text = f'<span style="color: {color};">{text}</span>'
                self.ui.resultBrowser.setHtml(formatted_text)
            if color == "red":
                formatted_text = f'<span style="color: {color};">{text}</span>'
                self.ui.resultBrowser.setHtml(formatted_text)
        elif name == "evalResult":
            self.ui.evalBrowser.setText(text)
        elif name == "log":
            append_text(self.ui.logBrowser, text)

    def update_tab_status(self, name):
        tab_index = None

        if name == "input":
            tab_index = self.ui.tabWidget.indexOf(self.ui.inputTab)
        elif name == "mask":
            tab_index = self.ui.tabWidget.indexOf(self.ui.maskTab)
        elif name == "eval":
            tab_index = self.ui.tabWidget.indexOf(self.ui.evalTab)
        elif name == "enhanced":
            tab_index = self.ui.tabWidget.indexOf(self.ui.enhancedTab)
        elif name == "masked":
            tab_index = self.ui.tabWidget.indexOf(self.ui.maskedTab)
        elif name == "current":
            tab_index = self.ui.tabWidget.indexOf(self.ui.currentTab)

        if tab_index is not None:
            # Set the stylesheet for the selected tab
            self.ui.tabWidget.tabBar().setTabTextColor(tab_index, QColor('blue'))

            # Apply a stylesheet to change the background color of the selected tab
            self.ui.tabWidget.setStyleSheet("""
                QTabBar::tab:selected {
                    background: lightgreen;
                }
                QTabBar::tab {
                    background: none;
                }
            """)

            # Ensure the selected tab is updated correctly
            self.ui.tabWidget.setCurrentIndex(tab_index)

    def update_feature_table(self, features):
        self.ui.tableWidget.setRowCount(len(features) + 1)
        for row, (feature, value) in enumerate(features.items()):
            self.ui.tableWidget.setItem(row, 0, QTableWidgetItem(feature))
            self.ui.tableWidget.setItem(row, 1, QTableWidgetItem(str(value)))

    def add_importance(self, importance_str):
        # Parse the importance string
        importance_dict = {}
        for line in importance_str.split('\n'):
            if ':' in line:
                feature, value = line.split(':')
                feature = feature.strip()
                try:
                    # Extract the feature number, removing the 'x' prefix
                    feature_num = int(feature[1:])  # Extract 4 from 'x4'
                    importance_value = float(value.strip())
                    importance_dict[feature_num] = importance_value
                except ValueError:
                    continue

        # Update the table widget with the parsed importance values
        for feature_num, importance_value in importance_dict.items():
            # Ensure the row number exists; note that row numbers start from 0, so subtract 1
            if feature_num - 1 < self.ui.tableWidget.rowCount():
                # Create a new QTableWidgetItem and set the text to the formatted importance value
                item = QTableWidgetItem(f"{importance_value:.4f}")
                # Set the item in the corresponding row (feature_num - 1) and the third column (index 2)
                self.ui.tableWidget.setItem(feature_num - 1, 2, item)

    def set_cell_color(self, color_labels):
        # Assuming self.tableWidget is the QTableWidget instance
        row_count = self.ui.tableWidget.rowCount()

        # Iterate over color_labels and set the corresponding cell color
        for i, color_label in enumerate(color_labels):
            # Make sure we do not exceed the row count
            if i < row_count:

                # Get the item at the specified cell
                item = self.ui.tableWidget.item(i, 1)  # Second column, row i

                # Set the background color based on the color_label
                if color_label == "green":
                    color = QColor(144, 238, 144)
                elif color_label == "yellow":
                    color = QColor(255, 255, 153)
                elif color_label == "red":
                    color = QColor(255, 182, 193)
                else:
                    color = QColor(255, 255, 255)

                item.setBackground(QBrush(color))

    def update_progressBar(self, progress):

        self.ui.progressBar.setValue(progress)
