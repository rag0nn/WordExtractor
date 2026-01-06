import cv2
import numpy as np
from engine import Engine

# TEST
### static
def prepare(pth):
    image = cv2.imread(pth)
    return image
    
def show(imgs,h_count = 3, frame_size = (1540,720)):
    part_count = len(imgs) // h_count
    if  len(imgs) % h_count != 0:
        part_count += 1
        for i in range(len(imgs) % h_count+1):
            imgs.append(np.zeros_like(imgs[-1]))
    
    mon_imgs = []
    for i in range(part_count):
        ims = imgs[i * h_count:i * h_count + h_count]
        mon =  np.hstack([cv2.cvtColor(_im,cv2.COLOR_GRAY2BGR) if (len(_im.shape) == 2) else _im for _im in ims])
        mon_imgs.append(mon) 
    
    for k, im in enumerate(mon_imgs):
        im = cv2.resize(im,frame_size)
        cv2.namedWindow(f"{k}",cv2.WINDOW_NORMAL)
        # cv2.setWindowProperty(f"{k}", cv2.WND_PROP_ASPECT_RATIO,cv2.WINDOW_FULLSCREEN)
        cv2.imshow(f"{k}",im)
    
    cv2.waitKey(0)
    
### Changeful
E = Engine()

def test(img):
    imgs = E.apply(img)
    return imgs

paths = [
    r"C:\Users\asus\rag0nn\EngReader\test_data\1.JPG",
    # r"C:\Users\asus\rag0nn\EngReader\test_data\2.JPG",
    # r"C:\Users\asus\rag0nn\EngReader\test_data\3.jpeg",
    # r"C:\Users\asus\rag0nn\EngReader\test_data\z5.jpeg",
    ]

for pth in paths:
    im = prepare(pth)
    res_seq = test(im)
    show(res_seq,3)


        
