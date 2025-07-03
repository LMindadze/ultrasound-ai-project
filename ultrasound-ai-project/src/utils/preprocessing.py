import cv2
import numpy as np

def extract_roi(image_path):
    """
    Load an ultrasound image, threshold it to find the largest
    region (the organ), and crop to that bounding box.
    Fallback to center-cropping if no contour is found.
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Otsu threshold
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological opening to remove small noise
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(opening,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        # pad box
        pad = 10
        x, y = max(x - pad, 0), max(y - pad, 0)
        w = min(w + 2*pad, img.shape[1] - x)
        h = min(h + 2*pad, img.shape[0] - y)
        roi = img[y:y+h, x:x+w]
    else:
        # fallback: center square crop
        h_img, w_img = img.shape[:2]
        m = min(h_img, w_img)
        cx, cy = w_img//2, h_img//2
        roi = img[cy-m//2:cy+m//2, cx-m//2:cx+m//2]

    return roi
