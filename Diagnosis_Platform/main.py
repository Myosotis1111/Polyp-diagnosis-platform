"""
-------------------------------------------------
Project Name:
Author: YANG, Xinchen
Last Modified Date: 08/09/2024
Description：As the software contribution of Master's project:
             Enhancing Medical Diagnosis using Deep-learning based Image Segmentation
-------------------------------------------------
"""

from view import MainView
from PySide2.QtWidgets import QApplication

app = QApplication([])
main = MainView()
main.ui.show()
app.exec_()
