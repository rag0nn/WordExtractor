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
    
    # padding
    if  len(imgs) % h_count != 0:
        part_count += 1
        padding_count = h_count - (len(imgs) % h_count)
        print("Padding count: ", padding_count)
        for i in range(padding_count):
            imgs.append(np.zeros_like(imgs[-1]))
    
    print("part count: ", part_count)
    print("len images", len(imgs))
    
    # segment
    mon_imgs = []
    for i in range(part_count):
        ims = imgs[i * h_count:i * h_count + h_count]
        # print("Segment length: ",len(ims))
        # for im in ims:
        #     print("Sub image shape: ", im.shape)
        
        mon =  np.hstack([cv2.cvtColor(_im,cv2.COLOR_GRAY2BGR) if (len(_im.shape) == 2) else _im for _im in ims])
        # print("Mon im shape: ", mon.shape)
        
        mon_imgs.append(mon) 
    
    # prepare to show and show
    for k, im in enumerate(mon_imgs):
        im = cv2.resize(im,frame_size)
        print(f"MON {k} SHAPE: ", im.shape)
        cv2.namedWindow(f"{k}",cv2.WINDOW_NORMAL)
        cv2.imshow(f"{k}",im)
    
    cv2.waitKey(0)
    
### Changeful
E = Engine()

def test(img):
    result_imgs, step_out_images = E.apply(img)
    for im in step_out_images:
        print(im.shape)
    return step_out_images

paths = [
    # r"C:\Users\asus\rag0nn\EngReader\test_data\1.JPG",
    # r"C:\Users\asus\rag0nn\EngReader\test_data\2.JPG",
    # r"C:\Users\asus\rag0nn\EngReader\test_data\3.jpeg",
    # r"C:\Users\asus\rag0nn\EngReader\test_data\5.jpeg",
    
    # r"C:\Users\asus\rag0nn\EngReader\raw_data\1.jpeg",
    # r"C:\Users\asus\rag0nn\EngReader\raw_data\2.jpeg",
    # r"C:\Users\asus\rag0nn\EngReader\raw_data\3.jpeg",
    # r"C:\Users\asus\rag0nn\EngReader\raw_data\5.jpeg",
    
    r"C:\Users\asus\rag0nn\EngReader\test_data\s-0.jpg",
    # r"C:\Users\asus\rag0nn\EngReader\test_data\s-1.jpg",
    # r"C:\Users\asus\rag0nn\EngReader\test_data\s-2.jpg",
    # r"C:\Users\asus\rag0nn\EngReader\test_data\s-3.jpg",q
    # r"C:\Users\asus\rag0nn\EngReader\test_data\s-4.jpg",
    
    
    # r"C:\Users\asus\rag0nn\EngReader\test_data\t-5.jpg",
    # r"C:\Users\asus\rag0nn\EngReader\test_data\t-6.jpg",
    # r"C:\Users\asus\rag0nn\EngReader\test_data\t-7.jpg",
    # r"C:\Users\asus\rag0nn\EngReader\test_data\t-8.jpg",
    # r"C:\Users\asus\rag0nn\EngReader\test_data\t-9.jpg",
    # r"C:\Users\asus\rag0nn\EngReader\test_data\t-10.jpg",
    # r"C:\Users\asus\rag0nn\EngReader\test_data\t-11.jpg",
    # r"C:\Users\asus\rag0nn\EngReader\test_data\t-12.jpg",
    
    # r"C:\Users\asus\rag0nn\EngReader\test_data\t-0.jpg",
    # r"C:\Users\asus\rag0nn\EngReader\test_data\t-1.jpg",
    # r"C:\Users\asus\rag0nn\EngReader\test_data\t-2.jpg",
    # r"C:\Users\asus\rag0nn\EngReader\test_data\t-3.jpg",
    # r"C:\Users\asus\rag0nn\EngReader\test_data\t-4.jpg",
    
    ]

for pth in paths:
    im = prepare(pth)
    res_seq = test(im)
    show(res_seq,3)


        
