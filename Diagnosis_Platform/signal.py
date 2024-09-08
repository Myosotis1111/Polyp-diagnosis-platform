from PySide2.QtCore import Signal, QObject, QDateTime
from PySide2.QtGui import QPixmap


class MySignals(QObject):

    update_image = Signal(QPixmap, str)

    update_text = Signal(str, str, str)

    update_tab_status = Signal(str)

    update_feature_table = Signal(dict)

    add_importance = Signal(str)

    set_cell_color = Signal(list)

    update_processBar = Signal(int)