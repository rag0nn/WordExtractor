import cv2
import numpy as np

# Görüntüyü oku
img = cv2.imread(r"C:\Users\asus\rag0nn\EngReader\inp.jpg")
orig = img.copy()

# Gri
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Adaptif threshold (dokümanlar için ideal)
thresh = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY_INV,
    15,
    5
)

# Yatay çizgiler için kernel
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))

# Sadece yatay çizgileri çıkar
lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# Contour'ları bul
contours, _ = cv2.findContours(
    lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

underline_hulls = []

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    area = cv2.contourArea(cnt)

    # Filtreler (çok önemli)
    if area < 200:
        continue
    if w / h < 5:  # yatay olmalı
        continue

    # Convex Hull
    hull = cv2.convexHull(cnt)
    underline_hulls.append(hull)

    # Çiz
    cv2.drawContours(orig, [hull], -1, (0, 0, 255), 2)

    # Hull koordinatları
    coords = hull.reshape(-1, 2)
    print("Alt çizgi koordinatları:")
    print(coords)
    print("-" * 30)

# Göster
cv2.imshow("Underlines with Convex Hull", cv2.resize(orig,(600,1200)))
cv2.imshow("lines", cv2.resize(lines,(600,1200)))

cv2.waitKey(0)
cv2.destroyAllWindows()


print(lines.shape)
