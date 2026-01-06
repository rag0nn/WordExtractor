#!/usr/bin/env python3
"""
EngReader GUI - PyQt6 arayüzü
Kullanım: python main.py
"""

from gui import ReaderGui
from PyQt6.QtWidgets import QApplication
import sys

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = ReaderGui()
    sys.exit(app.exec())
