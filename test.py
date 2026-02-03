# from testbase.tester import *
import cv2
# test_im(r"C:\Users\asus\rag0nn\EngReader\data\s-0.jpg",["fixation","skew"],True)
from engine import Engine

path = r"C:\Users\asus\Desktop\WhatsApp Image 2026-01-20 at 03.31.54.jpeg"

img = cv2.imread(path)


E = Engine()

a,b = E.apply(img)


cv2.imwrite("./result.jpg",a[1])