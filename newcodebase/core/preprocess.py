import cv2
import numpy as np
import pytesseract
import re
import logging

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
        logging.debug("Dörtgen çok bozuk, minAreaRect'e düşülüyor")
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
        logging.debug("Doc is None")
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
    logging.debug("Tesseract OSD:\n", osd)

    angle = int(re.search(r"Rotate: (\d+)", osd).group(1))
    logging.debug("Tesseract OSD Angle:", angle)

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

    logging.debug("ProjectProfile Skew Angle: ", best_angle)
    return deskewed