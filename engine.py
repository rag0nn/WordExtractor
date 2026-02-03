import cv2
import pytesseract
import numpy as np
from deep_translator import GoogleTranslator
import json
import os
from pathlib import Path
import logging
from PIL import Image, ImageDraw, ImageFont
import re

class _EngineSteps:
    
    kernel1 = np.array([
        [-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1],
        [-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1],
        [-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1],
        [-0.1,0.2,0.3,0.6,0.8,1,1,1,0.8,0.6,0.3,0.2,-0.1],
        [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
        [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
        [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
    ], dtype=np.float32)
    
    kernel2 = np.array([
        [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
        [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
        [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
        [-0.1,0.2,0.3,0.6,0.8,1,1,1,0.8,0.6,0.3,0.2,-0.1],
        [-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1],
        [-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1],
        [-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1],
    ], dtype=np.float32)
   
    kernel3 = np.array([
        [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,],
        [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,],
        [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,],
        [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,],
        [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,],
        [-0.5,-0.2,-0.1,0.6,0.8,1,1,1,1,1,1,1,1,1,0.8,0.6,-0.1,-0.2,-0.5],
        [-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,],
        [-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,],
        [-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,],
        [-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,],
        [-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,],
    ], dtype=np.float32)
     
    kernel4 = np.array([
        [-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,],
        [-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,],
        [-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,],
        [-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,],
        [-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,],
        [-0.5,-0.2,-0.1,0.6,0.8,1,1,1,1,1,1,1,1,1,0.8,0.6,-0.1,-0.2,-0.5],
        [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,],
        [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,],
        [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,],
        [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,],
        [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,],

    ], dtype=np.float32)
    kernel5 = np.array([
        [-0.5,-0.5,-0.5],
        [-0.5,-0.5,-0.5],
        [-0.5,-0.5,-0.5],
        [-0.5,-0.5,-0.5],
        [-0.5,-0.5,-0.5],
        [0,0,0],
        [0,0,0],
        [0,0,0],
        [0.1,0.1,0.1],
        [0.1,0.1,0.1],
        [0.1,0.1,0.1],
        [0.1,0.1,0.1],
        [0.1,0.1,0.1],
        [0.1,0.1,0.1],
        [0.1,0.1,0.1],
        [0,0,0],
        [0,0,0],
        [0,0,0],
        [-0.5,-0.5,-0.5],
        [-0.5,-0.5,-0.5],
        [-0.5,-0.5,-0.5],
        [-0.5,-0.5,-0.5],
        [-0.5,-0.5,-0.5],
    ], dtype=np.float32)    
    kernel6 = np.array([
        [-0.5,-0.5,-0.5],
        [-0.5,-0.5,-0.5],
        [-0.5,-0.5,-0.5],
        [-0.5,-0.5,-0.5],
        [0.1,0.1,0.1],
        [0.1,0.1,0.1],
        [0.1,0.1,0.1],
        [0.1,0.1,0.1],
        [0.1,0.1,0.1],
        [0.1,0.1,0.1],
        [0.1,0.1,0.1],
        [0.1,0.1,0.1],
        [0.1,0.1,0.1],
        [0.1,0.1,0.1],
        [0.1,0.1,0.1],
        [-0.5,-0.5,-0.5],
        [-0.5,-0.5,-0.5],
        [-0.5,-0.5,-0.5],
        [-0.5,-0.5,-0.5],
    ], dtype=np.float32)
    kernelX = np.array([
        # [,,,,,,,,,,,,,,,,,,,,,,]
        [0,0,0,0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0,0,0,0],
        [0,0,0,0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0,0,0,0],
        [0,0,0,0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0,0,0,0],
        [0,0,0,0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0,0,0,0],
        [0,0,0,0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0,0,0,0],
        [0,0,0,0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0,0,0,0],
        [0,0,0,0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0,0,0,0],
        [0,0,0,0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0,0,0,0],
        [0,0,0,0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0,0,0,0],
        [0,0,0,0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0,0,0,0],
        [0,0,0,0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0,0,0,0],
        [0,0,0,0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0,0,0,0],
        [0,0,0,0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0,0,0,0],
        [0,0,0,0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0,0,0,0],
        [0,0,0,0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0,0,0,0],
        [0,0,0,0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0,0,0,0],
        [0,0,0,0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0,0,0,0],
        [0,0,0,0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0,0,0,0],
        [0.1,0.1,0.1,0.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1,0.1,0.1,0.1],
        [0.1,0.1,0.1,0.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1,0.1,0.1,0.1],
        [0.1,0.1,0.1,0.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1,0.1,0.1,0.1],
        [0.1,0.1,0.1,0.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1,0.1,0.1,0.1],
        [0.1,0.1,0.1,0.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1,0.1,0.1,0.1],
        [0.1,0.1,0.1,0.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1,0.1,0.1,0.1],
        [0.1,0.1,0.1,0.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1,0.1,0.1,0.1],
        [0,0,0,0,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,0,0,0,0],
        [0,0,0,0,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,0,0,0,0],
        [0,0,0,0,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,0,0,0,0],
        [0,0,0,0,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,0,0,0,0],
        [0,0,0,0,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,0,0,0,0],
        [0,0,0,0,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,0,0,0,0],
        [0,0,0,0,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,0,0,0,0],
        [0,0,0,0,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,0,0,0,0],
        [0,0,0,0,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,0,0,0,0],
        [0,0,0,0,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,0,0,0,0],
        [0,0,0,0,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,0,0,0,0],
        [0,0,0,0,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,0,0,0,0],
        [0,0,0,0,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,0,0,0,0],
        [0,0,0,0,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,0,0,0,0],
        [0,0,0,0,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,0,0,0,0],
        [0,0,0,0,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,0,0,0,0],
        [0,0,0,0,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,0,0,0,0]

    ], dtype=np.float32)
    
    RATIOHW = 1.4
    # WIDTH = 1240
    # HEIGHT = 1754
    WIDTH = 2480
    HEIGHT = 3508
    @staticmethod
    def fixation(img:cv2.Mat)->cv2.Mat:
        orig_height,orig_width = img.shape[:2]
        
        summary = img.copy()
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        
        edges = cv2.Canny(blur, 50, 150)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
        edges_filled = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        edges_filled = cv2.dilate(edges_filled, kernel, 2)

        contours, _ = cv2.findContours(
            edges_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        edgeimg = edges.copy()
        fillededgeimg = edges_filled.copy()
        
        cntim = img.copy()
        if len(edgeimg.shape) == 2:
            edgeimg = cv2.cvtColor(edgeimg,cv2.COLOR_GRAY2BGR)
        if len(fillededgeimg.shape) == 2:
            fillededgeimg = cv2.cvtColor(fillededgeimg,cv2.COLOR_GRAY2BGR)
        if len(cntim.shape) == 2:
            cntim = cv2.cvtColor(cntim,cv2.COLOR_GRAY2BGR)
        cv2.drawContours(cntim,contours,-1,(255,200,0),2)
        
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        cnt = contours[0]

        cnt = cv2.convexHull(cnt)

        doc = None
        peri = cv2.arcLength(cnt, True)

        for eps in np.linspace(0.01, 0.05, 20):
            approx = cv2.approxPolyDP(cnt, eps * peri, True)
            if len(approx) == 4:
                doc = approx
                break

        if doc is None:
            rect = cv2.minAreaRect(cnt)
            doc = cv2.boxPoints(rect)
            doc = np.int8(doc)

        def angle(p1, p2, p3):
            v1 = p1 - p2
            v2 = p3 - p2
            return np.degrees(
                np.arccos(
                    np.dot(v1, v2) /
                    (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                )
            )

        angles = []
        pts = doc.reshape(4,2)
        for i in range(4):
            angles.append(angle(pts[i-1], pts[i], pts[(i+1)%4]))

        if not all(60 < a < 120 for a in angles):
            print("Dörtgen çok bozuk, minAreaRect'e düşülüyor")
            rect = cv2.minAreaRect(cnt)
            doc = cv2.boxPoints(rect)
            doc = np.int8(doc)


        def order_points(pts):
            pts = pts.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")

            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]     # top-left
            rect[2] = pts[np.argmax(s)]     # bottom-right

            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]  # top-right
            rect[3] = pts[np.argmax(diff)]  # bottom-left

            return rect
        
        summary = np.hstack([summary,edgeimg,fillededgeimg])
        
        if doc is None:
            print("Doc is None")
            return img, summary
        
        rect = order_points(doc)

        (w1, w2) = (
            np.linalg.norm(rect[0] - rect[1]),
            np.linalg.norm(rect[2] - rect[3])
        )
        h1 = np.linalg.norm(rect[0] - rect[3])
        h2 = np.linalg.norm(rect[1] - rect[2])

        width = int(max(w1, w2))
        height = int(max(h1, h2))

        dst = np.float32([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ])

        H = cv2.getPerspectiveTransform(rect, dst)

        fixed = cv2.warpPerspective(img, H, (width, height))

        summary = np.hstack([summary,cntim])
        
        summary = np.hstack([summary,cv2.resize(fixed, img.shape[::2][::-1])])
        
        fixed = cv2.resize(fixed,(orig_width,orig_height))
        
        doc = np.array(doc, dtype=np.int32).reshape(-1, 1, 2)
        
        cv2.drawContours(
            cntim,
            [doc],        # TEK contour olduğu için liste
            -1,
            (0, 255, 0),  # renk
            thickness=cv2.FILLED
        )
 
        return fixed,summary
    
    @staticmethod
    def skew_w_tesseract(img: cv2.Mat, out_ratio, show_summary:bool = False):
        osd = pytesseract.image_to_osd(
            img, output_type=pytesseract.Output.STRING
        )
        print("Tesseract OSD:\n", osd)

        angle = int(re.search(r"Rotate: (\d+)", osd).group(1))
        print("Tesseract OSD Angle:", angle)

        if angle == 0:
            return img.copy(), False

        h, w = img.shape[:2]

        angle_rad = np.deg2rad(angle)

        cos = abs(np.cos(angle_rad))
        sin = abs(np.sin(angle_rad))

        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)

        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2
        
        rotated = cv2.warpAffine(
            img,
            M,
            (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )

        # --- İçeriği crop et ---
        gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

        coords = cv2.findNonZero(th)
        x, y, w2, h2 = cv2.boundingRect(coords)

        cropped = rotated[y:y+h2, x:x+w2]
        
        h,w,c = img.shape
        new_w = int(h / out_ratio)
        cropped = cv2.resize(cropped,(new_w,h))
        # print("Cropped Shape: ", cropped.shape)
        
        if show_summary:      
            def _prepare_show(img):
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
                img = cv2.resize(img,(500,900))
                return img
            
            a1 = _prepare_show(img.copy())
            a2 = _prepare_show(rotated.copy())
            a3 = _prepare_show(th.copy())
            a4 = _prepare_show(cropped.copy())
            
            summary =np.hstack([a1,a2,a3,a4])
            summary = cv2.resize(summary,(1400,600))
            cv2.imshow("Tessaract Od Sumary", summary)
            cv2.waitKey(0)
            
        return cropped, True

    @staticmethod
    def skew_w_project_profile(img:cv2.Mat)->cv2.Mat:

        def projection_score(img):
            hist = np.sum(img, axis=1)
            return np.var(hist)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, bw = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        (h, w) = bw.shape
        center = (w // 2, h // 2)

        best_angle = 0
        best_score = -1
       
        # küçük açılar yeterlidir (belge için)
        for angle in np.arange(-5, 5, 0.2):
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                bw, M, (w, h),
                flags=cv2.INTER_NEAREST,
                borderValue=0
            )

            score = projection_score(rotated)

            if score > best_score:
                best_score = score
                best_angle = angle

        # Final rotate (orijinal görüntü)
        M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
        deskewed = cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

        print("ProjectProfile Skew Angle: ", best_angle)
        return deskewed

    @staticmethod
    def preprocess(img:cv2.Mat,width,height)->cv2.Mat:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        gray = cv2.resize(gray,(width,height))
        return gray
    
    @staticmethod
    def letterfilter(img,kernel):
        filtered = cv2.filter2D(img, cv2.CV_32F, kernel)
        filtered_norm = cv2.normalize(
            filtered,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX
        ).astype(np.uint8)
        return filtered_norm

    @staticmethod
    def blurFilter(img):
        img = cv2.GaussianBlur(img,(21,9),31,sigmaY=1)
        return img
    
    @staticmethod
    def saturationFilters(img):
        k = np.ones((5, 11), np.uint8)  # Define 5x5 kernel   
        morph1 = cv2.dilate(img, k, 1)
        # morph1 = cv2.dilate(morph1, k, 1)
        k = np.ones((1,33), np.uint8)  # Define 5x5 kernel   
        morph1 = cv2.dilate(morph1, k, 1)
           
        # ret, thresh = cv2.threshold(
        #     morph1, 0, 255,
        #     cv2.THRESH_BINARY + cv2.THRESH_OTSU
        # )
        ret, thresh = cv2.threshold(
            morph1, 190, 255,
            cv2.THRESH_BINARY
        )
        return thresh
    
    @staticmethod
    def extractionContours(img):
        contours,hierarchy = cv2.findContours(img, 1, 2)
        cnt_image = cv2.cvtColor(img.copy(),cv2.COLOR_GRAY2BGR)
        # if len(contours) > 0:
        #         cv2.drawContours(cnt_image, contours, -1, (255,120,0), 5)
        return cnt_image, contours
    
    # @staticmethod
    # def extractionRect(img, contours):
    #     boxes = []
    #     for cnt in contours:
    #         if len(cnt) < 3:
    #             continue   # geçersiz contour
            
    #         area = cv2.contourArea(cnt)
    #         if area > 1 and area < 10000:
    #             rect = cv2.minAreaRect(cnt)
    #             box = cv2.boxPoints(rect)
    #             box = np.int32(box)   
    #             boxes.append(box)
                
    #     boxes_image = cv2.cvtColor(img.copy(),cv2.COLOR_GRAY2BGR)
    #     if len(boxes) > 0:
    #         cv2.drawContours(boxes_image, boxes, -1, (0,0,255), 2)
            
    #     return boxes_image,boxes
    
    @staticmethod
    def extractionRect(img, contours):
        boxes = []      
        boxes_image = cv2.cvtColor(img.copy(),cv2.COLOR_GRAY2BGR)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 300 or area > 6000:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            # if len(approx) not in [4,5,6]:
            #     continue

            x, y, w, h = cv2.boundingRect(approx)
            ratio = w / h

            if ratio  > 3.0:
                boxes.append((x,y,w,h))
        
        print(f"Found {len(boxes)} boxes")
        for x,y,w,h in boxes:
            cv2.rectangle(boxes_image,(x,y),(x+w,y+h),(0,0,255),5)
        return boxes_image, boxes
    
    # @staticmethod
    # def extractionRect(binary):
    #     # -----------------------------
    #     # PARAMETRELER
    #     # -----------------------------
    #     KERNEL_W = 150
    #     KERNEL_H = 35

    #     MIN_FILL = 0.0001    # çok boş olmasın
    #     MAX_FILL = 0.95   # tamamen dolu olmasın (büyük dikdörtgeni ele)
    #     STRIDE = 4        # tarama adımı

    #     # -----------------------------
    #     # BINARY GÖRÜNTÜ (0 / 255)
    #     # -----------------------------
    #     binary = (binary > 0).astype(np.uint8)  # 0 / 1 yap

    #     H, W = binary.shape

    #     # -----------------------------
    #     # INTEGRAL IMAGE
    #     # -----------------------------
    #     ii = cv2.integral(binary)

    #     def rect_sum(ii, x, y, w, h):
    #         return ii[y+h, x+w] - ii[y, x+w] - ii[y+h, x] + ii[y, x]

    #     detections = []

    #     # -----------------------------
    #     # SLIDING WINDOW
    #     # -----------------------------
        
    #     fill_fail = 0
    #     edge_fail = 0
    #     for y in range(0, H - KERNEL_H, STRIDE):
    #         for x in range(0, W - KERNEL_W, STRIDE):

    #             # 1️⃣ İç doluluk
    #             filled = rect_sum(ii, x, y, KERNEL_W, KERNEL_H)
    #             area = KERNEL_W * KERNEL_H
    #             fill_ratio = filled / area

    #             if not (MIN_FILL < fill_ratio < MAX_FILL):
    #                 fill_fail += 1
    #                 continue

    #             # 2️⃣ Kenar kontrolü (tamamen içeride olmalı)
    #             top    = rect_sum(ii, x, y, KERNEL_W, 1)
    #             bottom = rect_sum(ii, x, y + KERNEL_H - 1, KERNEL_W, 1)
    #             left   = rect_sum(ii, x, y, 1, KERNEL_H)
    #             right  = rect_sum(ii, x + KERNEL_W - 1, y, 1, KERNEL_H)

    #             if top > 0 or bottom > 0 or left > 0 or right > 0:
    #                 edge_fail += 1
    #                 continue

    #             # 3️⃣ Kabul
    #             detections.append((x, y, KERNEL_W, KERNEL_H))

    #     vis = np.zeros_like(binary)
    #     vis = cv2.cvtColor(vis,cv2.COLOR_GRAY2BGR)
    #     for x, y, w, h in detections:
    #         cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 0, 255), 1)
            
    #     print("Bulunan Rect: ", len(detections))
    #     print("Fill fail: ", fill_fail)
    #     print("Edge fail: ", edge_fail)
    #     return vis, detections


class Engine:
    
    def __init__(self):
        pass
    
    def _print_process_step(self,img,text,coor=(50,200),color=(0,0,255)):
        if len(img.shape) == 2:
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        cv2.putText(img,text,coor,cv2.FONT_HERSHEY_DUPLEX,3.8,color,5)
        return img
    
    def skew(self,img):
        try:
            skewed_t, is_skewed = _EngineSteps.skew_w_tesseract(img, _EngineSteps.RATIOHW)
            skewed_p = _EngineSteps.skew_w_project_profile(skewed_t)  
        except Exception as e:
            print(f"Tessaract Deskew failed. {e}")
            skewed_t = np.zeros_like(img)
            
            h,w = None,None
            if(len(skewed_t.shape) == 3):
                h,w,c = skewed_t.shape
            else:
                h,w = skewed_t.shape
            h = int(h/2)
            w = int(w/2)
            cv2.putText(skewed_t,"Tesseract Failed",(w,h),cv2.FONT_HERSHEY_DUPLEX,1.5,(0,0,255),1)           
            skewed_p = _EngineSteps.skew_w_project_profile(img)

        so_skewed_p = self._print_process_step(skewed_p.copy(),"ProjectProfileSkew")
        so_skewed_t = self._print_process_step(skewed_t.copy(),"TesseractSkew")
        return skewed_p, so_skewed_p, so_skewed_t
    
    def fixation(self,img):
        img, summary = _EngineSteps.fixation(img)
        # summary = cv2.resize(summary,(1250,700))
        # cv2.imshow("Fixation summary", summary)
        # cv2.waitKey(0)
        
        so_fix = self._print_process_step(img.copy(),"Fixation")
        return img, so_fix
    
    def preprocess(self,img):
        out = _EngineSteps.preprocess(img,_EngineSteps.WIDTH,_EngineSteps.HEIGHT)    
        so_preprocess = self._print_process_step(out.copy(),"Preprocess")
        return out, so_preprocess

    def letterFilters(self,img):

        img = _EngineSteps.letterfilter(img,_EngineSteps.kernelX)

        img = _EngineSteps.letterfilter(img,_EngineSteps.kernel1)
        img = _EngineSteps.letterfilter(img,_EngineSteps.kernel2)
        img = _EngineSteps.letterfilter(img,_EngineSteps.kernel1)
        img = _EngineSteps.letterfilter(img,_EngineSteps.kernel2)
        img = _EngineSteps.letterfilter(img,_EngineSteps.kernel1)
        img = _EngineSteps.letterfilter(img,_EngineSteps.kernel2)
        img = _EngineSteps.letterfilter(img,_EngineSteps.kernel1)
        img = _EngineSteps.letterfilter(img,_EngineSteps.kernel3)
        img = _EngineSteps.letterfilter(img,_EngineSteps.kernel4)
        # img = _EngineSteps.letterfilter(img,_EngineSteps.kernelX)
        
        # # img = _EngineSteps.letterfilter(img,_EngineSteps.kernel5)
        # # img = _EngineSteps.letterfilter(img,_EngineSteps.kernel5)
        # img = _EngineSteps.letterfilter(img,_EngineSteps.kernel6)
        # img = _EngineSteps.letterfilter(img,_EngineSteps.kernel6)
        # img = _EngineSteps.letterfilter(img,_EngineSteps.kernel6)
        # img = _EngineSteps.letterfilter(img,_EngineSteps.kernel6)

        
        so_letter = self._print_process_step(img.copy(),"LetterFilter")
        return img, so_letter
    
    def blurFilter(self,img):
        img = _EngineSteps.blurFilter(img)
        so_blur = self._print_process_step(img.copy(),"BlurFilter")
        return img, so_blur
    
    def saturationFilters(self,img):
        saturationfiltered =  _EngineSteps.saturationFilters(img)
        so_satration = self._print_process_step(saturationfiltered,"Saturation")
        return saturationfiltered, so_satration
    
    def extractionContours(self,img):
        extractioncontorsim, contours = _EngineSteps.extractionContours(img)
        so_contour = self._print_process_step(extractioncontorsim.copy(),"Contour")
        return extractioncontorsim, contours, so_contour
    
    def extractionRect(self,img,contours):
        extractionboxesim, boxes = _EngineSteps.extractionRect(img,contours)
        so_extraction_rect = self._print_process_step(extractionboxesim,"Rect Extraction")
        return extractionboxesim, boxes, so_extraction_rect
    
    def apply(self,img):
        fixed, so_fixed = self.fixation(img)
        skewed,so_skewed_projectprofile, so_skewed_tesseract = self.skew(fixed)
        cv2.imwrite("./so_skewed.jpg",cv2.resize(skewed,(_EngineSteps.WIDTH,_EngineSteps.HEIGHT)))
        preprocessed, so_preprocesed = self.preprocess(skewed)
        cv2.imwrite("./so_preprocessed.jpg",cv2.cvtColor(preprocessed,cv2.COLOR_GRAY2BGR))
        letterfiltered, so_letter_filtered = self.letterFilters(preprocessed)
        # blurfiltered, so_blured = self.blurFilter(letterfiltered)
        saturationfiltered, so_saturated = self.saturationFilters(letterfiltered)
        extractioncontorsim, contours, so_contour = self.extractionContours(saturationfiltered)
        cv2.imwrite("./so_counter.jpg",so_contour)
        extractionboxesim, boxes, so_extraction_Rect = self.extractionRect(saturationfiltered,contours)
        
        main_pipe_result = [
            fixed,
            skewed,
            preprocessed,
            letterfiltered,
            # blurfiltered,
            saturationfiltered,
            extractioncontorsim,
            extractionboxesim
        ]
        step_output_result = [
            img,
            so_fixed,
            so_skewed_tesseract,
            so_skewed_projectprofile,
            so_preprocesed,
            so_letter_filtered,
            # so_blured,
            so_saturated,
            so_contour,
            so_extraction_Rect
        ]
        return main_pipe_result, step_output_result
    

                
        
