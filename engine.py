import cv2
import pytesseract
import numpy as np
from deep_translator import GoogleTranslator
import json
import os
from pathlib import Path
import logging
from PIL import Image, ImageDraw, ImageFont

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
    
    @staticmethod
    def preprocess(img:cv2.Mat)->cv2.Mat:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
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
    def saturationFilters(img):
        k = np.ones((5, 11), np.uint8)  # Define 5x5 kernel   
        morph1 = cv2.dilate(img, k, 1)
        # morph1 = cv2.dilate(morph1, k, 1)
        k = np.ones((1,33), np.uint8)  # Define 5x5 kernel   
        morph1 = cv2.dilate(morph1, k, 1)
           
        ret,thresh = cv2.threshold(morph1,100,255,cv2.THRESH_OTSU)
        return thresh
    
    @staticmethod
    def extractionContours(img):
        contours,hierarchy = cv2.findContours(img, 1, 2)
        cnt_image = cv2.cvtColor(img.copy(),cv2.COLOR_GRAY2BGR)
        if len(contours) > 0:
                cv2.drawContours(cnt_image, contours, -1, (0,0,255), 2)
        return cnt_image, contours
    
    @staticmethod
    def extractionRect(img, contours):
        boxes = []
        for cnt in contours:
            if len(cnt) < 3:
                continue   # geçersiz contour
            
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int32(box)   
            boxes.append(box)
            
        boxes_image = cv2.cvtColor(img.copy(),cv2.COLOR_GRAY2BGR)
        if len(boxes) > 0:
            cv2.drawContours(boxes_image, boxes, -1, (0,0,255), 2)
            
        return boxes_image,boxes



class Engine:
    
    def __init__(self):
        pass
    
    def preprocess(self,img):
        out = _EngineSteps.preprocess(img)    
        return out
    
    def letterFilters(self,img):
        img = _EngineSteps.letterfilter(img,_EngineSteps.kernel1)
        img = _EngineSteps.letterfilter(img,_EngineSteps.kernel2)
        img = _EngineSteps.letterfilter(img,_EngineSteps.kernel1)
        img = _EngineSteps.letterfilter(img,_EngineSteps.kernel2)
        img = _EngineSteps.letterfilter(img,_EngineSteps.kernel1)
        img = _EngineSteps.letterfilter(img,_EngineSteps.kernel2)
        img = _EngineSteps.letterfilter(img,_EngineSteps.kernel1)
        return img
    
    def saturationFilters(self,img):
        saturationfiltered =  _EngineSteps.saturationFilters(img)
        return saturationfiltered
    
    def extractionContours(self,img):
        extractioncontorsim, contours = _EngineSteps.extractionContours(img)
        return extractioncontorsim, contours
    
    def extractionRect(self,img,contours):
        extractionboxesim, boxes = _EngineSteps.extractionRect(img,contours)
        return extractionboxesim, boxes
    
    def apply(self,img):
        preprocessed = self.preprocess(img)
        letterfiltered = self.letterFilters(preprocessed)
        saturationfiltered = self.saturationFilters(letterfiltered)
        extractioncontorsim, contours = self.extractionContours(saturationfiltered)
        extractionboxesim, boxes = self.extractionRect(saturationfiltered,contours)
        return [img,preprocessed,letterfiltered,saturationfiltered,extractioncontorsim,extractionboxesim]
    

                
        
