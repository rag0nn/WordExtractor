from engine import Engine
import logging
import cv2

logging.basicConfig(level=logging.INFO)

def main():
    im_paths = [r"C:\Users\asus\rag0nn\EngReader\inp.jpg"]
    
    
    for idx, im_path in enumerate(im_paths):
        logging.info(f"=== [{idx}] {im_path} =========")
        engine = Engine()
        image = cv2.imread(im_path)
        out = engine.apply(image)
        steps_imgs = engine.get_steps_images(1600,700,3)
        # cv2.imshow("out",out)
        for i, step_img in enumerate(steps_imgs):
            cv2.imshow(f"{i}", step_img)
        waitkey = cv2.waitKey(0)
        if waitkey == ord("q"):
            continue

if __name__ == "__main__":
    main()

