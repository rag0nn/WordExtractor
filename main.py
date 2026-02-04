from engine import Engine
from core.elements import Saver
import logging
import cv2

logging.basicConfig(level=logging.INFO)

def main():
    im_paths = [r"C:\Users\asus\rag0nn\EngReader\inp.jpg"]
    
    
    for idx, im_path in enumerate(im_paths):
        logging.info(f"=== [{idx}] {im_path} =========")
        engine = Engine()
        image = cv2.imread(im_path)
        translated_words = engine.apply(image)
        saver = Saver(translated_words)
        out_pth = f"{im_path.split('/')[-1].split('.')[0]}.csv"
        saver.save(out_pth)
        logging.info(f"Saved {out_pth}")
        
        steps_imgs = engine.get_steps_images(1600,700,3)

        for i, step_img in enumerate(steps_imgs):
            cv2.imshow(f"{i}", step_img)
        waitkey = cv2.waitKey(0)
        if waitkey == ord("q"):
            continue

if __name__ == "__main__":
    main()

