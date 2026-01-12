import sys
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict
import json
import os
from datetime import datetime
import logging

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QScrollArea, QFileDialog, QMessageBox,
    QProgressBar, QTabWidget, QSlider, QSpinBox, QGroupBox, QGridLayout,
    QScrollBar
)
from PyQt6.QtGui import QPixmap, QImage, QFont, QIcon
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize

try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False

from engine import Engine
from structs.word import Word

logging.basicConfig(level=logging.INFO)

class ZoomableImageLabel(QLabel):
    """Zoom ve pan özelliği olan resim etiketi"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_pixmap = None
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.setStyleSheet("border: 1px solid #999; background-color: #f5f5f5;")
    
    def set_pixmap(self, pixmap: QPixmap):
        """Resmi ayarla"""
        self.original_pixmap = pixmap
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.update_display()
    
    def set_zoom(self, level: float):
        """Zoom seviyesini ayarla"""
        self.zoom_level = max(0.1, min(5.0, level))
        self.update_display()
    
    def set_pan(self, x: int, y: int):
        """Pan konumunu ayarla"""
        self.pan_x = x
        self.pan_y = y
        self.update_display()
    
    def update_display(self):
        """Gösterimi güncelle"""
        if self.original_pixmap is None:
            self.setText("Resim yok")
            return
        
        # Zoom uygula
        scaled_size = int(self.original_pixmap.width() * self.zoom_level)
        scaled_pixmap = self.original_pixmap.scaledToWidth(
            scaled_size,
            Qt.TransformationMode.SmoothTransformation
        )
        
        # Pan uygula (crop)
        label_width = self.width()
        label_height = self.height()
        
        if scaled_pixmap.width() > label_width or scaled_pixmap.height() > label_height:
            x = max(0, min(self.pan_x, scaled_pixmap.width() - label_width))
            y = max(0, min(self.pan_y, scaled_pixmap.height() - label_height))
            cropped = scaled_pixmap.copy(x, y, label_width, label_height)
            QLabel.setPixmap(self, cropped)
        else:
            QLabel.setPixmap(self, scaled_pixmap)
    
    def resizeEvent(self, event):
        """Resize olduğunda güncelle"""
        super().resizeEvent(event)
        if self.original_pixmap:
            self.update_display()
            
class ProcessThread(QThread):
    """Thread'te resim işleme işlemlerini gerçekleştir"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, engine:Engine, img_path:str):
        super().__init__()
        self.results = None
        self.engine = engine
        self.img_path = img_path

    def run(self):
        def _shape_convert(img):
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img
            return img

        try:
            img = cv2.imread(self.img_path)
            if img is None:
                raise Exception(f"Görüntü yüklenemiyor: {self.img_path}")
            
            fixed, so_fixed = self.engine.fixation(img)
            self.progress.emit(10)    
            skewed,so_skewed_projectprofile, so_skewed_tesseract = self.engine.skew(fixed)
            self.progress.emit(20)    
            preprocessed, so_preprocesed = self.engine.preprocess(skewed)
            self.progress.emit(30)    
            letterfiltered, so_letter_filtered = self.engine.letterFilters(preprocessed)
            self.progress.emit(40)    
            blurfiltered, so_blured = self.engine.blurFilter(letterfiltered)
            self.progress.emit(50)    
            saturationfiltered, so_saturated = self.engine.saturationFilters(blurfiltered)
            self.progress.emit(60)    
            extractioncontorsim, contours, so_contour = self.engine.extractionContours(saturationfiltered)
            self.progress.emit(80)    
            extractionboxesim, boxes, so_extraction_Rect = self.engine.extractionRect(saturationfiltered,contours)
            self.progress.emit(100)    
                 
            so_fixed = _shape_convert(so_fixed)
            so_skewed_tesseract = _shape_convert(so_skewed_tesseract)
            so_skewed_projectprofile = _shape_convert(so_skewed_projectprofile)
            process = np.hstack([_shape_convert(im) for im in [
                so_fixed,
                so_skewed_tesseract, 
                so_skewed_projectprofile,
                so_preprocesed,
                so_letter_filtered,
                so_blured,
                so_saturated,
                so_contour,
                so_extraction_Rect
                ]])

            self.results = {
                'path': self.img_path,
                'original': img,
                'process': process,
                'result': extractionboxesim,
            }
            
            self.finished.emit([self.results])
        except Exception as e:
            self.error.emit(str(e))
            
class ImageThumbnail(QPushButton):
    """Resim thumbnail butonu"""
    def __init__(self, image_path: str, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.setFixedSize(100, 100)
        self.setFlat(True)
        self.is_selected = False
        
        # Thumbnail görseli yükle
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaledToWidth(100, Qt.TransformationMode.SmoothTransformation)
        self.setIcon(QIcon(pixmap))
        self.setIconSize(QSize(100, 100))
        
        # Tooltip olarak dosya adını göster
        self.setToolTip(Path(image_path).name)
    
    def set_selected(self, selected: bool):
        """Seçim durumunu ayarla ve stil uygula"""
        self.is_selected = selected
        if selected:
            # Seçili - turkuaz arka plan
            self.setStyleSheet("""
                QPushButton {
                    background-color: #6BA3A3;
                    border: 3px solid #2C3E50;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #5A8E8E;
                }
            """)
        else:
            # Seçilmemiş - transparan
            self.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    border: 1px solid #ccc;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #E8F4F8;
                }
            """)



class ReaderGui(QMainWindow):
    # Soft renk şeması
    COLORS = {
        'primary': '#E8F4F8',      # Açık mavi-yeşil
        'secondary': '#F5F5F5',    # Açık gri
        'accent': '#6BA3A3',       # Turkuaz
        'text_dark': '#2C3E50',    # Koyu mavi-gri
        'text_light': '#FFFFFF',   # Beyaz
        'success': '#A8D5BA',      # Açık yeşil
        'warning': '#FFE8B6',      # Açık turuncu
        'button': '#B8E6D5',       # Mint yeşil
    }

    def __init__(self):
        super().__init__()
        
        # Config'i oku
        self.config = self.load_config()
        
        self.engine = Engine()
        self.image_paths = []
        self.process_results = []
        self.current_image_index = 0
        self.thumbnail_buttons = []
        
        self.init_ui()
        self.apply_stylesheet()

    def load_config(self):
        """Config dosyasını oku"""
        try:
            conf_path = str(Path(os.getcwd(), "config.json"))
            with open(conf_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Config yüklenemiyor: {e}")
            return {}

    def apply_stylesheet(self):
        """Uygulama genelinde stil ayarla"""
        stylesheet = f"""
        QMainWindow, QWidget {{
            background-color: {self.COLORS['secondary']};
            color: {self.COLORS['text_dark']};
        }}
        
        QPushButton {{
            background-color: {self.COLORS['button']};
            color: {self.COLORS['text_dark']};
            border: none;
            border-radius: 4px;
            padding: 8px 12px;
            font-weight: bold;
            font-size: 11px;
        }}
        
        QPushButton:hover {{
            background-color: {self.COLORS['accent']};
            color: {self.COLORS['text_light']};
        }}
        
        QPushButton:pressed {{
            background-color: #5A8E8E;
        }}
        
        QLabel {{
            color: {self.COLORS['text_dark']};
        }}
        
        QGroupBox {{
            border: 2px solid {self.COLORS['accent']};
            border-radius: 6px;
            margin-top: 6px;
            padding-top: 6px;
            color: {self.COLORS['text_dark']};
            font-weight: bold;
        }}
        
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px 0 3px;
        }}
        
        QSlider::groove:horizontal {{
            border: 1px solid {self.COLORS['accent']};
            height: 8px;
            background: {self.COLORS['primary']};
            border-radius: 4px;
        }}
        
        QSlider::handle:horizontal {{
            background: {self.COLORS['accent']};
            border: 2px solid {self.COLORS['accent']};
            width: 18px;
            margin: -5px 0;
            border-radius: 9px;
        }}
        
        QSpinBox {{
            background-color: {self.COLORS['primary']};
            border: 1px solid {self.COLORS['accent']};
            border-radius: 4px;
            padding: 4px;
            color: {self.COLORS['text_dark']};
        }}
        
        QProgressBar {{
            border: 2px solid {self.COLORS['accent']};
            border-radius: 4px;
            text-align: center;
            color: {self.COLORS['text_dark']};
        }}
        
        QProgressBar::chunk {{
            background-color: {self.COLORS['success']};
            border-radius: 2px;
        }}
        
        QTabBar::tab {{
            background-color: {self.COLORS['primary']};
            color: {self.COLORS['text_dark']};
            padding: 6px 20px;
            border: 1px solid {self.COLORS['accent']};
            border-bottom: none;
        }}
        
        QTabBar::tab:selected {{
            background-color: {self.COLORS['accent']};
            color: {self.COLORS['text_light']};
        }}
        
        QTabWidget::pane {{
            border: 1px solid {self.COLORS['accent']};
        }}
        
        QScrollArea {{
            border: 1px solid {self.COLORS['accent']};
            background-color: {self.COLORS['primary']};
        }}
        """
        self.setStyleSheet(stylesheet)

    def init_ui(self):
        """GUI'yi başlat"""
        self.setWindowTitle("Reader - OCR & Translation Tool")
        self.setGeometry(100, 100, 1600, 900)
        
        # Ana widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        main_layout = QHBoxLayout()
        
        # SOL PANEL - Resim Listesi
        left_panel = self.create_left_panel()
        main_layout.addLayout(left_panel, 1)
        
        # SAĞ PANEL - Görsel Gösterimi ve Kontroller
        right_panel = self.create_right_panel()
        main_layout.addLayout(right_panel, 3)
        
        main_widget.setLayout(main_layout)
        
        self.show()

    def create_left_panel(self) -> QVBoxLayout:
        """Sol panel: Resim listesi ve kontrol butonları"""
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Başlık
        title = QLabel("📷 Images")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Resim thumbnail'leri için scroll area
        self.thumbnail_scroll = QScrollArea()
        self.thumbnail_scroll.setWidgetResizable(True)
        self.thumbnail_scroll.setStyleSheet(f"""
            QScrollArea {{
                background-color: {self.COLORS['primary']};
                border: 2px solid {self.COLORS['accent']};
                border-radius: 4px;
            }}
        """)
        
        self.thumbnail_container = QWidget()
        self.thumbnail_layout = QVBoxLayout()
        self.thumbnail_layout.setSpacing(5)
        self.thumbnail_layout.addStretch()
        self.thumbnail_container.setLayout(self.thumbnail_layout)
        
        self.thumbnail_scroll.setWidget(self.thumbnail_container)
        layout.addWidget(self.thumbnail_scroll, 1)
        
        # Kontrol Butonları
        button_layout = QVBoxLayout()
        button_layout.setSpacing(8)
        
        self.import_btn = QPushButton("➕ Add Image")
        self.import_btn.clicked.connect(self.import_images)
        self.import_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        button_layout.addWidget(self.import_btn)
        
        self.process_single_btn = QPushButton("▶ Execute Single")
        self.process_single_btn.clicked.connect(self.process_single)
        self.process_single_btn.setEnabled(False)
        self.process_single_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        button_layout.addWidget(self.process_single_btn)
        
        self.process_batch_btn = QPushButton("⏭ Execute Batch")
        self.process_batch_btn.clicked.connect(self.process_batch)
        self.process_batch_btn.setEnabled(False)
        self.process_batch_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        button_layout.addWidget(self.process_batch_btn)
        
        self.export_single_btn = QPushButton("💾 Export Single")
        self.export_single_btn.clicked.connect(self.export_single)
        self.export_single_btn.setEnabled(False)
        self.export_single_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        button_layout.addWidget(self.export_single_btn)
        
        self.export_batch_btn = QPushButton("💾 Export Batch")
        self.export_batch_btn.clicked.connect(self.export_batch)
        self.export_batch_btn.setEnabled(False)
        self.export_batch_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        button_layout.addWidget(self.export_batch_btn)
        
        self.clear_btn = QPushButton("🗑 Clear")
        self.clear_btn.clicked.connect(self.clear_all)
        self.clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        button_layout.addWidget(self.clear_btn)
        
        layout.addLayout(button_layout)
        
        return layout

    def create_right_panel(self) -> QVBoxLayout:
        """Sağ panel: Seçili görselin işlenmiş hallerini göster ve kontroller"""
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Başlık ve bilgi
        header_layout = QHBoxLayout()
        title = QLabel("👁 Image Flow")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        header_layout.addWidget(title)
        
        self.status_label = QLabel("")
        header_layout.addStretch()
        header_layout.addWidget(self.status_label)
        
        layout.addLayout(header_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Görsel gösterimi için tab widget
        self.display_tabs = QTabWidget()
        
        # Original sekme
        original_tab = QScrollArea()
        original_tab.setWidgetResizable(True)
        self.original_label = ZoomableImageLabel()
        original_tab.setWidget(self.original_label)
        self.display_tabs.addTab(original_tab, "📌 Original")
        
        # Preprocessed sekme
        process_tab = QScrollArea()
        process_tab.setWidgetResizable(True)
        self.process_label = ZoomableImageLabel()
        process_tab.setWidget(self.process_label)
        self.display_tabs.addTab(process_tab, "🔬 Process")
        
        # Result sekme (Gerçek işleme sonucu)
        result_tab = QScrollArea()
        result_tab.setWidgetResizable(True)
        self.result_label = ZoomableImageLabel()
        result_tab.setWidget(self.result_label)
        self.display_tabs.addTab(result_tab, "✅ Result")
        
        layout.addWidget(self.display_tabs, 2)
        
        # Çıktı bilgileri
        self.output_info = QLabel("Any process executed")
        self.output_info.setStyleSheet(f"padding: 10px; background-color: {self.COLORS['primary']}; border-radius: 4px; border: 1px solid {self.COLORS['accent']};")
        layout.addWidget(self.output_info)
        
        return layout

    def import_images(self):
        """Resim içeri aktar"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Resim Seç",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff);;All Files (*)"
        )
        
        if file_paths:
            self.image_paths.extend(file_paths)
            # process_results listesini yeni resimler için extend et
            self.process_results.extend([None] * len(file_paths))
            self.update_thumbnails()
            self.process_single_btn.setEnabled(True)
            self.process_batch_btn.setEnabled(True)
            
    def update_thumbnails(self):
        """Thumbnail'leri güncelle"""
        # Önceki thumbnail'leri kaldır (stretch hariç)
        while self.thumbnail_layout.count() > 1:
            item = self.thumbnail_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Yeni thumbnail'ler ekle
        self.thumbnail_buttons = []
        for idx, img_path in enumerate(self.image_paths):
            thumb = ImageThumbnail(img_path)
            thumb.clicked.connect(lambda checked, i=idx: self.select_image(i))
            self.thumbnail_layout.insertWidget(idx, thumb)
            self.thumbnail_buttons.append(thumb)
        
        # İlk resmi seç
        if self.image_paths:
            self.select_image(0)
                
    def select_image(self, index: int):
        """Resim seç ve göster"""
        self.current_image_index = index
        
        # Thumbnail'leri güncelle - seçili olanı vurgula
        if hasattr(self, 'thumbnail_buttons'):
            for idx, thumb in enumerate(self.thumbnail_buttons):
                thumb.set_selected(idx == index)
        
        # Mevcut işlenmiş sonuç varsa göster
        has_result = (
            self.process_results and 
            index < len(self.process_results) and 
            self.process_results[index] is not None
        )
        
        if has_result:
            result = self.process_results[index]
            self.display_processed_image(result)
        else:
            # İşlenmiş görsel yoksa sadece orijinal resmi göster
            self.display_original_image(index)
        
    def process_single(self):
        """Seçili resmi işle"""
        if self.current_image_index < len(self.image_paths):
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            img_path = self.image_paths[self.current_image_index]
            
            self.process_thread = ProcessThread(
                self.engine,
                img_path,
            )
            self.process_thread.progress.connect(self.progress_bar.setValue)
            self.process_thread.finished.connect(self.on_process_finished)
            self.process_thread.error.connect(self.on_process_error)
            self.process_thread.start()
                
    def on_process_finished(self, results):
        """İşleme tamamlandığında"""
        # Sonucu mevcut index'e ekle (results bir liste, [0] indeksi dict'tir)
        self.process_results[self.current_image_index] = results[0]
        
        self.progress_bar.setVisible(False)
        self.export_single_btn.setEnabled(True)
        self.export_batch_btn.setEnabled(True)
        
        # Seçili resmi göster (işlenmiş sonuç gösterilecek)
        self.select_image(self.current_image_index)
        
        QMessageBox.information(self, "✅ Done", f"Image processed successfully.")

    def on_process_error(self, error_msg: str):
        """İşleme hatası oluştuğunda"""
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "❌ Failed", f"Process Fault: {error_msg}")
        
    def display_processed_image(self, result: Dict):
        """İşlenmiş resim ve sonuçları göster"""
        img_path = result['path']
        
        # Orijinal resmi göster
        img = result['original']
        self.show_image_in_label(img, self.original_label)
        
        # Preprocessed resmi göster
        if 'process' in result:
            process = result['process']
            self.show_image_in_label(process, self.process_label)
        
        # Result resmi göster (gerçek işleme sonucu - çizilmiş)
        if 'result' in result:
            result_img = result['result']
            self.show_image_in_label(result_img, self.result_label)
        
        filename = Path(img_path).name
        self.status_label.setText(f"Seçili: {filename} ({self.current_image_index + 1}/{len(self.image_paths)}) - ✅ İşlendi")
        self.output_info.setText("İşlem tamamlandı. Tüm sekmeler mevcuttur.")
    
    def display_original_image(self, index: int):
        """Seçili orijinal resmi göster (işlenmemiş)"""
        if index < len(self.image_paths):
            img_path = self.image_paths[index]
            img = cv2.imread(img_path)
            
            if img is not None:
                # Orijinal resmi göster
                self.show_image_in_label(img, self.original_label)
                
                # Process ve Result sekmelerini temizle
                self.process_label.set_pixmap(QPixmap())
                self.result_label.set_pixmap(QPixmap())
                
                filename = Path(img_path).name
                self.status_label.setText(f"Seçili: {filename} ({index + 1}/{len(self.image_paths)}) - İşlenmemiş")
                self.output_info.setText("Henüz işlem yapılmamış. 'Execute Single' butonuna tıklayarak işlemi başlatın.")
            else:
                QMessageBox.warning(self, "⚠️ Hata", f"Resim yüklenemiyor: {img_path}")
  
    def show_image_in_label(self, cv_image, label: ZoomableImageLabel):
        """OpenCV görselini ZoomableImageLabel'e göster"""
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        label.set_pixmap(pixmap)
    
    def process_batch(self):
        """Tüm görselleri batch işle"""
        if not self.image_paths:
            QMessageBox.warning(self, "⚠️ Uyarı", "İşlemek için resim seçilmemiş.")
            return
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.process_batch_btn.setEnabled(False)
        
        # Batch işleme başlat
        self.batch_process_index = 0
        self._process_batch_next()
    
    def _process_batch_next(self):
        """Batch'teki bir sonraki resmi işle"""
        if self.batch_process_index < len(self.image_paths):
            # Önce resmi seç
            self.select_image(self.batch_process_index)
            
            # Sonra işle
            img_path = self.image_paths[self.batch_process_index]
            
            self.process_thread = ProcessThread(self.engine, img_path)
            self.process_thread.progress.connect(self.progress_bar.setValue)
            self.process_thread.finished.connect(self._on_batch_process_finished)
            self.process_thread.error.connect(self.on_process_error)
            self.process_thread.start()
        else:
            # Batch tamamlandı
            self.progress_bar.setVisible(False)
            self.process_batch_btn.setEnabled(True)
            self.export_batch_btn.setEnabled(True)
            QMessageBox.information(self, "✅ Tamamlandı", f"Tüm {len(self.image_paths)} resim işlendi.")
    
    def _on_batch_process_finished(self, results):
        """Batch işlemde bir resim tamamlandığında"""
        # Sonucu kaydet (results bir liste, [0] indeksi dict'tir)
        self.process_results[self.batch_process_index] = results[0]
        
        # İlerleme güncelle
        self.batch_process_index += 1
        progress = int((self.batch_process_index / len(self.image_paths)) * 100)
        self.progress_bar.setValue(progress)
        
        # Bir sonraki resmi işle
        self._process_batch_next()
    
    def export_single(self):
        """Seçili resmi ve işleme sonuçlarını klasöre kaydет"""
        if self.current_image_index >= len(self.image_paths):
            QMessageBox.warning(self, "⚠️ Hata", "Geçerli bir resim seçilmemiş.")
            return
        
        # Hedef klasör seç
        target_dir = QFileDialog.getExistingDirectory(
            self,
            "Kaydedilecek Klasörü Seç",
            ""
        )
        
        if not target_dir:
            return
        
        # Seçili resmin adını al (uzantı hariç)
        img_path = self.image_paths[self.current_image_index]
        img_name = Path(img_path).stem
        
        # Resim adıyla yeni klasör oluştur
        export_dir = Path(target_dir) / img_name
        export_dir.mkdir(exist_ok=True)
        
        # İşlenmiş sonuç var mı kontrol et
        has_result = (
            self.process_results and 
            self.current_image_index < len(self.process_results) and 
            self.process_results[self.current_image_index] is not None
        )
        
        try:
            if has_result:
                result = self.process_results[self.current_image_index]
                
                # Original kaydет
                original_path = export_dir / f"{img_name}_original.png"
                cv2.imwrite(str(original_path), result['original'])
                
                # Process kaydет
                process_path = export_dir / f"{img_name}_process.png"
                cv2.imwrite(str(process_path), result['process'])
                
                # Result kaydет
                result_path = export_dir / f"{img_name}_result.png"
                cv2.imwrite(str(result_path), result['result'])
                
                QMessageBox.information(
                    self, 
                    "✅ Kaydedildi", 
                    f"Resim başarıyla kaydedildi:\n{export_dir}"
                )
            else:
                # Sadece orijinal kaydет
                img = cv2.imread(img_path)
                original_path = export_dir / f"{img_name}_original.png"
                cv2.imwrite(str(original_path), img)
                
                QMessageBox.information(
                    self, 
                    "ℹ️ Bilgi", 
                    f"Sadece orijinal resim kaydedildi (işlenmiş sonuç yok):\n{export_dir}"
                )
        
        except Exception as e:
            QMessageBox.critical(self, "❌ Hata", f"Kaydederken hata oluştu: {str(e)}")
    
    def export_batch(self):
        """Tüm görselleri batch olarak klasörlere kaydет"""
        if not self.image_paths:
            QMessageBox.warning(self, "⚠️ Hata", "Kaydedilecek resim yok.")
            return
        
        # Hedef klasör seçimi sadece bir kez
        target_dir = QFileDialog.getExistingDirectory(
            self,
            "Tüm Dosyaları Kaydedilecek Ana Klasörü Seç",
            ""
        )
        
        if not target_dir:
            return
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.export_batch_btn.setEnabled(False)
        
        success_count = 0
        
        try:
            for idx, img_path in enumerate(self.image_paths):
                img_name = Path(img_path).stem
                export_dir = Path(target_dir) / img_name
                export_dir.mkdir(exist_ok=True)
                
                # İşlenmiş sonuç var mı kontrol et
                has_result = (
                    self.process_results and 
                    idx < len(self.process_results) and 
                    self.process_results[idx] is not None
                )
                
                if has_result:
                    result = self.process_results[idx]
                    
                    # Original kaydет
                    cv2.imwrite(str(export_dir / f"{img_name}_original.png"), result['original'])
                    
                    # Process kaydет
                    cv2.imwrite(str(export_dir / f"{img_name}_process.png"), result['process'])
                    
                    # Result kaydет
                    cv2.imwrite(str(export_dir / f"{img_name}_result.png"), result['result'])
                else:
                    # Sadece orijinal kaydет
                    img = cv2.imread(img_path)
                    cv2.imwrite(str(export_dir / f"{img_name}_original.png"), img)
                
                success_count += 1
                progress = int(((idx + 1) / len(self.image_paths)) * 100)
                self.progress_bar.setValue(progress)
            
            self.progress_bar.setVisible(False)
            self.export_batch_btn.setEnabled(True)
            
            QMessageBox.information(
                self, 
                "✅ Tamamlandı", 
                f"{success_count} resim başarıyla kaydedildi:\n{target_dir}"
            )
        
        except Exception as e:
            self.progress_bar.setVisible(False)
            self.export_batch_btn.setEnabled(True)
            QMessageBox.critical(self, "❌ Hata", f"Batch kaydederken hata oluştu: {str(e)}")
    
    def clear_all(self):
        """Tüm görselleri ve sonuçları temizle"""
        if not self.image_paths:
            QMessageBox.information(self, "ℹ️ Bilgi", "Temizlenecek resim yok.")
            return
        
        reply = QMessageBox.question(
            self,
            "❓ Onay",
            "Tüm görselleri ve işleme sonuçlarını temizlemek istediğinizden emin misiniz?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Thumbnail'leri kaldır
            while self.thumbnail_layout.count() > 1:
                item = self.thumbnail_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            
            # Listeleri temizle
            self.image_paths = []
            self.process_results = []
            self.thumbnail_buttons = []
            self.current_image_index = 0
            
            # UI temizle
            self.original_label.set_pixmap(QPixmap())
            self.process_label.set_pixmap(QPixmap())
            self.result_label.set_pixmap(QPixmap())
            self.output_info.setText("Resim yok")
            self.status_label.setText("Durum: Hazır")
            
            # Butonları devre dışı bırak
            self.process_single_btn.setEnabled(False)
            self.process_batch_btn.setEnabled(False)
            self.export_single_btn.setEnabled(False)
            self.export_batch_btn.setEnabled(False)
            
            QMessageBox.information(self, "✅ Temizlendi", "Tüm görseller başarıyla temizlendi.")
        

import sys

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = ReaderGui()
    sys.exit(app.exec())
