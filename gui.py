import sys
import os
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QListWidget, QTabWidget, QTextEdit, QFileDialog, 
                             QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QSplitter,
                             QLabel, QFrame, QMessageBox)
from PyQt6.QtCore import Qt, pyqtSignal, QRectF
from PyQt6.QtGui import QImage, QPixmap, QPainter, QAction, QColor, QFont

from engine import Engine
from core.elements import Saver

class ImageViewer(QGraphicsView):
    # Signal to sync zoom/pan with other viewers
    view_changed = pyqtSignal(object, object, object) # matrix, h_val, v_val

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.image_item = None
        self._is_syncing = False

    def set_image(self, cv_img):
        self.scene.clear()
        if cv_img is None:
            return

        # Convert cv2 image to QPixmap
        if len(cv_img.shape) == 2:
            height, width = cv_img.shape
            bytes_per_line = width
            q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
        else:
            height, width, channel = cv_img.shape
            bytes_per_line = 3 * width
            # OpenCV is BGR, PyQt is RGB
            cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            q_img = QImage(cv_img_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)

        self.image_item = QGraphicsPixmapItem(QPixmap.fromImage(q_img))
        self.scene.addItem(self.image_item)
        self.setSceneRect(QRectF(self.image_item.pixmap().rect()))
        self.fitInView(self.scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event):
        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor

        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor

        self.scale(zoom_factor, zoom_factor)
        self.notify_sync()

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.notify_sync()

    def notify_sync(self):
        # Emit current transform info to sync other views
        if not self._is_syncing:
            self.view_changed.emit(self.transform(), self.horizontalScrollBar().value(), self.verticalScrollBar().value())

    def sync_state(self, transform, h_val, v_val):
        self._is_syncing = True
        self.setTransform(transform)
        self.horizontalScrollBar().setValue(h_val)
        self.verticalScrollBar().setValue(v_val)
        self._is_syncing = False

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Visual Processing App")
        self.resize(1200, 800)

        # Helper objects
        self.engine = None
        self.loaded_images = {} # path -> image_data
        self.processed_results = {} # path -> (words, saver)
        self.current_image_path = None
        self.viewers = [] # Keep track of all viewers for syncing

        self.setup_ui()
        self.apply_styles()

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # --- Left Bar (1/5 width) ---
        left_layout = QVBoxLayout()
        left_container = QWidget()
        left_container.setLayout(left_layout)
        left_container.setFixedWidth(250) # Approx 1/5 of 1200, better to specificy fixed or stretch

        self.btn_import = QPushButton("IMPORT IMAGES")
        self.btn_import.setFixedHeight(50)
        self.btn_import.clicked.connect(self.import_images)
        self.btn_import.setCursor(Qt.CursorShape.PointingHandCursor)

        self.list_images = QListWidget()
        self.list_images.itemClicked.connect(self.image_selected)

        self.btn_export = QPushButton("EXPORT ALL (CSV)")
        self.btn_export.setFixedHeight(50)
        self.btn_export.clicked.connect(self.export_data)
        self.btn_export.setCursor(Qt.CursorShape.PointingHandCursor)

        left_layout.addWidget(self.btn_import)
        left_layout.addWidget(self.list_images)
        left_layout.addWidget(self.btn_export)

        # --- Right Bar ---
        right_layout = QVBoxLayout()
        right_container = QWidget()
        right_container.setLayout(right_layout)

        # Top Bar (Tabs + Process Button)
        top_bar_layout = QHBoxLayout()
        self.tabs = QTabWidget()
        self.tabs.currentChanged.connect(self.on_tab_changed)
        
        self.btn_process = QPushButton("PROCESS")
        self.btn_process.setFixedSize(120, 40)
        self.btn_process.clicked.connect(self.process_current_image)
        self.btn_process.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_process.setEnabled(False)

        top_bar_layout.addWidget(self.tabs)
        top_bar_layout.addWidget(self.btn_process)

        # Bottom Right (Output Text)
        self.text_output = QListWidget()
        self.text_output.setFixedHeight(150)

        # Splitter to separate Image Area and Text Output
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # We need a container for the tabs to put in splitter
        tabs_container = QWidget()
        tabs_layout = QVBoxLayout(tabs_container)
        tabs_layout.setContentsMargins(0,0,0,0)
        tabs_layout.addLayout(top_bar_layout) # Put tabs and button in a layout

        right_splitter.addWidget(tabs_container)
        right_splitter.addWidget(self.text_output)
        right_splitter.setStretchFactor(0, 4)
        right_splitter.setStretchFactor(1, 1)

        right_layout.addWidget(right_splitter)

        # Add to main layout
        main_layout.addWidget(left_container)
        main_layout.addWidget(right_container)

    def apply_styles(self):
        # Vibrant, High Contrast Theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e2e;
            }
            QWidget {
                color: #cdd6f4;
                font-family: 'Segoe UI', sans-serif;
                font-size: 14px;
            }
            QPushButton {
                background-color: #89b4fa;
                color: #1e1e2e;
                border-radius: 8px;
                font-weight: bold;
                border: none;
            }
            QPushButton:hover {
                background-color: #b4befe;
            }
            QPushButton:pressed {
                background-color: #74c7ec;
            }
            QListWidget {
                background-color: #313244;
                border: 2px solid #45475a;
                border-radius: 8px;
                padding: 5px;
            }
            QListWidget::item {
                padding: 8px;
                border-radius: 4px;
            }
            QListWidget::item:selected {
                background-color: #f38ba8;
                color: #1e1e2e;
                font-weight: bold;
            }
            QTabWidget::pane {
                border: 2px solid #45475a;
                background-color: #313244;
                border-radius: 8px;
            }
            QTabBar::tab {
                background: #45475a;
                color: #cdd6f4;
                padding: 10px 20px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                margin-right: 2px; 
            }
            QTabBar::tab:selected {
                background: #f9e2af;
                color: #1e1e2e;
                font-weight: bold;
            }
            QTabBar::tab:hover {
                background: #585b70;
            }
        """)
        self.btn_export.setStyleSheet("""
            QPushButton {
                background-color: #a6e3a1; 
                color: #1e1e2e;
                font-weight: bold;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #94e2d5;
            }
        """)

    def import_images(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if files:
            for f in files:
                if f not in self.loaded_images:
                    self.loaded_images[f] = None # Will load on demand or now? Let's load on demand
                    item_name = os.path.basename(f)
                    self.list_images.addItem(item_name)
                    # Store full path in user role or just use index mapping if list doesn't change order
                    # Better: rely on self.loaded_images keys order matching list? No, dict is ordered in py3.7+, but let's be safe
                    # Just find path by text is risky if dup names.
                    # Let's simple store path in a separate list or handle it simply for now.
                    # Actually, we can use the item data.
                    item = self.list_images.item(self.list_images.count()-1)
                    item.setData(Qt.ItemDataRole.UserRole, f)
    
    def image_selected(self, item):
        path = item.data(Qt.ItemDataRole.UserRole)
        self.current_image_path = path
        
        # Load image if not loaded
        if self.loaded_images[path] is None:
             self.loaded_images[path] = cv2.imread(path)
        
        img = self.loaded_images[path]
        
        # Reset Tabs
        self.tabs.clear()
        self.viewers.clear()
        self.text_output.clear()
        
        # Enable Process Button
        self.btn_process.setEnabled(True)

        # Add Original Tab
        self.add_tab_image(img, "Original")

        # If already processed, show result tabs? 
        # Requirement says "tıklandığında resim seçme tabı gelecek" and "işle butonu Engine.apply".
        # So we probably reset state or check if cached.
        if path in self.processed_results:
            self.restore_processed_view(path)

    def add_tab_image(self, img, title):
        viewer = ImageViewer()
        viewer.set_image(img)
        viewer.view_changed.connect(self.sync_all_views)
        self.tabs.addTab(viewer, title)
        self.viewers.append(viewer)

    def process_current_image(self):
        if not self.current_image_path:
            return

        # Initialize Engine
        # Creating new engine instance per process as per main.py example structure inside loop?
        # Or reuse? engine=Engine() is lightweight usually.
        # main.py does: engine = Engine() inside loop.
        self.engine = Engine() # Re-init to clear internal state like self.process_images
        
        img = self.loaded_images[self.current_image_path]
        
        # Run apply
        try:
            # This returns 'translated_words'
            translated_words = self.engine.apply(img)
            
            # Save results for export and display
            saver = Saver(translated_words)
            self.processed_results[self.current_image_path] = (translated_words, saver)
            
            # Display text output
            self.text_output.clear()
            for w in translated_words:
                 self.text_output.addItem(f"{w.word} -> {w.equivalent}")

            # Get steps images
            # main.py calls get_steps_images(1600, 700, 3) 
            # We want one image per tab. So we might not want the tiled canvas return.
            # But Engine.get_steps_images returns canvases.
            # Wait, user requirement: "Engine.get_steps_images'dan dönen belirsiz sayıdaki görsel için birer tane [tab]".
            # If get_steps_images returns tiled canvases, maybe we should access engine.process_images directly?
            # User says: "Tab'lar olacak Engine.get_steps_images'dan dönen...".
            # Let's stick to what engine provides.
            step_imgs = self.engine.get_steps_images(1600, 900, 1) # Force 1 column to get roughly individual slides if possible?
            # logic in get_steps_images: chunks by column_count. If i pass column_count=1, it makes one canvas per image?
            # Let's check engine.py logic.
            # for start in range(0, total, column_count): chunk = ...
            # canvas size is fixed.
            # If I want raw images I should access self.engine.process_images directly if allowed.
            # Engine.process_images is public.
            # Logic in Engine.apply: appends img, then steps append to process_images.
            # Requirement explicitly mentions `Engine.get_steps_images`.
            # If I use `get_steps_images(800, 600, 1)`, it will create canvases with 1 image each, resized to fit.
            # I will use that as requested.
            
            steps = self.engine.get_steps_images(1200, 675, 1) # use a reasonable resolution
            
            # Add tabs (skip 0 because it's usually original which we already have? 
            # Engine.apply does: process_images.append(img) (Original)
            # Then step decorators append step_img.
            # So the list contains [Original, Step1, Step2, ...].
            # We already have Original tab. We can limit or just overwrite.
            # Le'ts clear generic tabs and re-add all from steps to be safe and consistent.
            
            current_zoom = self.viewers[0].transform() if self.viewers else None
            
            self.tabs.clear()
            self.viewers.clear()
            
            for i, step_img in enumerate(steps):
                title = f"Step {i}"
                # Try to extract name from image text if possible? No easy way.
                self.add_tab_image(step_img, title)
                
            # If we had a previous zoom, apply it?
            # Or just let them sync from scratch. 
            
        except Exception as e:
            print(f"Error processing image: {e}")
            import traceback
            traceback.print_exc()

    def restore_processed_view(self, path):
        # If we come back to a processed image, we should ideally show the results again.
        # But we don't store the step images in `processed_results`.
        # So we might need to re-run or store them.
        # For simplicity, let's just show original and if they click process it runs again (fast enough?)
        # Or store it.
        # Given "memory" constraints, re-running might be safer unless user wants persistence.
        # But 'processed_results' stores words for Export.
        
        words, saver = self.processed_results[path]
        self.text_output.clear()
        for w in words:
            self.text_output.addItem(f"{w.word} -> {w.equivalent}")
            
    def sync_all_views(self, transform, h_val, v_val):
        # Warning: avoid infinite loops.
        sender = self.sender()
        for viewer in self.viewers:
            if viewer != sender:
                # transform() returns QTransform which is passed directly
                viewer.sync_state(transform, h_val, v_val)

    # Correction in sync_all_views due to recursive signal potential
    # Handled by _is_syncing check in ImageViewer.notify_sync()

    def on_tab_changed(self, index):
        # Sync the new tab to the old tab's zoom?
        # If we have shared state, when we switch tab, the new tab should match the LAST active view state.
        # But current implementation syncs actively.
        # If I zoom in Tab1, Tab2 is updated in background.
        # So when I switch to Tab2, it is already zoomed.
        pass

    def export_data(self):
        # "işlenmiş tüm görsellerdeki çıktıları birer birer saver.save ile csv olarak kaydedecek"
        if not self.processed_results:
            QMessageBox.warning(self, "Export Failed", "No processed data to export.")
            return
            
        # Maybe ask for a folder? Or save next to original images?
        # User said "saver.save ile csv olarak kaydedecek", main.py saves to same folder.
        # We'll follow main.py pattern: "path/to/img.csv"
        
        count = 0
        errors = []
        for path, (words, saver) in self.processed_results.items():
            out_pth = f"{path.split('.')[0]}.csv"
            try:
                saver.save(out_pth)
                count += 1
            except Exception as e:
                print(f"Failed to save {out_pth}: {e}")
                errors.append(f"{os.path.basename(path)}: {str(e)}")
        
        if count > 0:
            msg = f"Successfully exported {count} files."
            if errors:
                msg += f"\n\nErrors occurred ({len(errors)}):\n" + "\n".join(errors)
                QMessageBox.warning(self, "Export Partial Success", msg)
            else:
                QMessageBox.information(self, "Export Success", msg)
        else:
             QMessageBox.critical(self, "Export Failed", f"Failed to export files.\nErrors:\n" + "\n".join(errors))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
