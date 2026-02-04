import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import re
import logging

from typing import List
from .elements import Line, Word

def extract_lines(img):
    orig = img.copy()
    # Gri
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Adaptif threshold
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        5
    )
    # Yatay çizgiler için kernel
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))

    # Sadece yatay çizgileri çıkar
    lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Contour'ları bul
    contours, _ = cv2.findContours(
        lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    lines = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)

        # Filtreler (çok önemli)
        if area < 200:
            continue
        if w / h < 3:  # yatay olmalı
            continue
        if w > 1000 or w < 50:
            continue
        # Convex Hull
        hull = cv2.convexHull(cnt)

        # Çiz
        cv2.drawContours(orig, [hull], -1, (0, 0, 255), 2)

        # Hull koordinatları
        coords = hull.reshape(-1, 2)
        logging.debug("Alt çizgi koordinatları:")
        logging.debug(coords)
        logging.debug("-" * 30)
        lines.append(Line(x,y,w,h))
       
    for k, line in enumerate(lines):
        cv2.putText(orig,f"{k}",(line.x,line.y),cv2.FONT_HERSHEY_DUPLEX,2.5,(255,0,0),5)
    logging.info(f"Found {len(lines)} line")
    return (orig,lines)

def expand_lines(lines: List[Line], y_range=50, x_range=80) -> List[Line]:
    if lines is None:
        return []

    if isinstance(lines, np.ndarray):
        if lines.size == 0:
            return []
        lines = list(lines)   # 🔑 çok önemli

    # 1️⃣ y'ye göre sırala
    lines = sorted(lines, key=lambda l: l.y)

    merged_lines: List[Line] = []
    used = [False] * len(lines)

    for i, base in enumerate(lines):
        if used[i]:
            continue

        cur_x = base.x
        cur_y = base.y
        cur_h = base.h
        cur_right = base.x + base.w

        used[i] = True

        # 2️⃣ aynı satırda zincirleme birleştir
        for j in range(i + 1, len(lines)):
            other = lines[j]

            if used[j]:
                continue

            # y kontrolü
            if abs(other.y - cur_y) > y_range:
                break  # y sıralı olduğu için daha aşağılar tutmaz

            # x yakınlık kontrolü
            if abs(other.x - cur_right) <= x_range or abs((other.x + other.w) - cur_x) <= x_range:
                cur_x = min(cur_x, other.x)
                cur_right = max(cur_right, other.x + other.w)
                cur_h = max(cur_h, other.h)
                used[j] = True

        merged_lines.append(
            Line(
                x=cur_x,
                y=cur_y,
                w=cur_right - cur_x,
                h=cur_h
            )
        )

    logging.info(f"Expanded lines: {len(merged_lines)}")
    return merged_lines
            
def extract_words(img,lines)->List[List[Word]]:
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    x_extra = 20
    
    segment_coords = []
    for line in lines:
        x1 = line.x-x_extra
        x2 = line.x+line.w+x_extra
        y1 = line.y+line.h-70
        y2 = line.y+line.h+10
        segment_coords.append((x1,x2,y1,y2))
        mask[y1:y2,x1:x2] = 255

    img = cv2.bitwise_and(img, img, mask=mask)
    
    words = []
        
    for line, segment_coor in zip(lines, segment_coords):
        line:Line
        
        # determine segment
        x_extra = 20
        segment_x1,segment_x2,segment_y1,segment_y2 = segment_coor
        roi = img[segment_y1:segment_y2,segment_x1:segment_x2]
        data = pytesseract.image_to_data(
            roi,
            lang="eng",
            output_type=Output.DICT
        )

        # find segment words
        segment_words = []
        for i in range(len(data["text"])):
            word = data["text"][i].strip()
            conf = int(data["conf"][i])

            if word != "" and conf > 25:   # confidence filtresi
                x = data["left"][i]
                y = data["top"][i]
                w = data["width"][i]
                h = data["height"][i]
                segment_words.append((word,conf,x,y,w,h))
                
        # concat segment words
        if segment_words :
            segment_word = " ".join(list( word[0] for word in segment_words))
            segment_words = segment_words[::-1]
            segment_x = min(list(word[2] for word in segment_words))
            segment_y = segment_words[0][3]
            segment_w = sum(list(wrd[4] for wrd in segment_words))
            segment_conf = sum(list( wrd[1] for wrd in segment_words)) / len(segment_words)
            words.append(Word(segment_word,segment_conf,line.x+segment_x-x_extra,line.y+segment_y-70,segment_w,h,0))
            
    return words, img
