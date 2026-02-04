import yaml
import os
import numpy as np
import cv2
import math
import logging
from typing import List
import time
from deep_translator import GoogleTranslator

import core as pcore
from core.elements import Line, Word

def _step_register(name):
    def decorator(func):
        def wrap(self, *args, **kwargs):
            start = time.perf_counter()

            logging.info(f"Execution: {name} ...")

            res_img = func(self, *args, **kwargs)

            elapsed_ms = (time.perf_counter() - start) * 1000

            logging.info(
                f"Execution: {name} Ended. "
                f"Elapsed: {elapsed_ms:.2f} ms"
            )

            step_img = res_img.copy()
            if len(step_img.shape) == 2:
                step_img = cv2.cvtColor(step_img, cv2.COLOR_GRAY2BGR)

            cv2.putText(
                step_img,
                # f"{name} | {elapsed_ms:.1f} ms",
                f"{name}",
                (50, 350),
                cv2.FONT_HERSHEY_DUPLEX,
                4.6,
                (0, 0, 255),
                3
            )

            self.process_images.append(step_img)
            return res_img

        return wrap
    return decorator

     
     

class Engine:
    
    def __init__(self):
        self.process_images = []
        self.config = None
        try:
            with open(os.path.dirname(__file__)+ "/" + "config.yaml") as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            raise f"Config did'nt load: {e}"
    
    def get_steps_images(self, canvas_w: int, canvas_h: int, column_count: int):
        """
        canvas_w, canvas_h : Her canvas'ın boyutu
        column_count       : Bir canvas'taki maksimum görsel sayısı (tek satır)
        self.process_images: List[np.ndarray]

        return: List[np.ndarray]
        """

        if self.process_images is None or len(self.process_images) == 0:
            return []

        images = self.process_images
        total_images = len(images)

        channels = 1 if images[0].ndim == 2 else images[0].shape[2]
        dtype = images[0].dtype

        canvases = []

        cell_w = canvas_w // column_count
        cell_h = canvas_h  # 🔑 tek row olduğu için tüm yükseklik

        for start in range(0, total_images, column_count):
            chunk = images[start : start + column_count]

            if channels == 1:
                canvas = np.zeros((canvas_h, canvas_w), dtype=dtype)
            else:
                canvas = np.zeros((canvas_h, canvas_w, channels), dtype=dtype)

            for col, img in enumerate(chunk):
                ih, iw = img.shape[:2]

                # aspect ratio korunur
                scale = min(cell_w / iw, cell_h / ih)
                new_w = int(iw * scale)
                new_h = int(ih * scale)

                resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

                x0 = col * cell_w
                y0 = 0

                x_offset = (cell_w - new_w) // 2
                y_offset = (cell_h - new_h) // 2

                canvas[
                    y0 + y_offset : y0 + y_offset + new_h,
                    x0 + x_offset : x0 + x_offset + new_w
                ] = resized

            canvases.append(canvas)

        return canvases

    @_step_register(name="skew")
    def skew(self,img):
        try:
            ratioWH = self.config["image_width"] / self.config["image_height"]
            skewed_t, is_skewed = pcore.skew_w_tesseract(img, ratioWH)
            skewed_p = pcore.skew_w_project_profile(skewed_t)  
        except Exception as e:
            logging.error(f"Tessaract Deskew failed. {e}")
            skewed_t = np.zeros_like(img)
            
            h,w = None,None
            if(len(skewed_t.shape) == 3):
                h,w,c = skewed_t.shape
            else:
                h,w = skewed_t.shape
            h = int(h/2)
            w = int(w/2)
            cv2.putText(skewed_t,"Tesseract Failed",(w,h),cv2.FONT_HERSHEY_DUPLEX,1.5,(0,0,255),1)           
            skewed_p = pcore.skew_w_project_profile(img)

        return skewed_p
    
    @_step_register(name='lines')
    def lines(self,img):
        out, lines = pcore.extract_lines(img)
        lines:List[Line]
        self.lines = lines
        return out
    
    @_step_register(name='expand_lines')
    def expand_lines(self,img,lines):
        newlines = pcore.expand_lines(lines)
        result = img.copy()
        for line in newlines:
            cv2.rectangle(result,(line.x,line.y),(line.x+line.w,line.y+line.h),(0,255,0),5)
        self.newlines = newlines
        return result
    
    @_step_register(name="extract_words")
    def extract_words(self,img,lines):
        words, masked_out = pcore.extract_words(img,lines)
        for wrd in words:
            logging.info(f"{wrd}")
            cv2.rectangle(masked_out, (wrd.x, wrd.y), (wrd.x+wrd.w, wrd.y+wrd.h-40), (0, 255, 0), 5)
            cv2.putText(masked_out, wrd.word, (wrd.x, wrd.y+wrd.h-40),
                        cv2.FONT_HERSHEY_DUPLEX, 1.4, (255,70,0), 4)
        self.extracted_words = words
        logging.info(f"Extracted Words : {len(words)}")
        return masked_out
    
    @_step_register(name='translate_words')
    def translate_words(self,img,words,src='en',target='tr'):
        out = img.copy()
        for wrd in words:
            wrd:Word
            wrd.equivalent = GoogleTranslator(source=src, target=target).translate(wrd.word)
            logging.info(f"{wrd.word} --> {wrd.equivalent}")
            cv2.rectangle(out, (wrd.x, wrd.y), (wrd.x+wrd.w, wrd.y+wrd.h), (0, 255, 0), 5)
            cv2.putText(out, wrd.equivalent, (wrd.x, wrd.y+wrd.h-5),
                        cv2.FONT_HERSHEY_DUPLEX, 1.4, (255,100,0), 4)
        self.translated_worlds = words
        return out

    def apply(self, img):
        self.process_images.clear()
        self.process_images.append(img)
        skewed = self.skew(img)        
        resized = cv2.resize(skewed,(self.config["image_height"],self.config["image_width"]))
        lines = self.lines(resized)
        expanded_lines = self.expand_lines(resized,self.lines)
        words_extracted = self.extract_words(resized,self.newlines)
        words_translated = self.translate_words(resized,self.extracted_words)
        cv2.imwrite("./output.jpg",words_translated)
        return self.translated_worlds
        