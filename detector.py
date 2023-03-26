import cv2
from skimage.metrics import structural_similarity
import numpy as np



class AnomalyDetector:
    def __init__(self):
        self.hist_size = 0
        self.history_imgs = []
        self.history_dates = []
        self.history_areas = []

    def add_to_history(self, img, coords=None, date=None):
        self.history_imgs.append(img)
        self.history_areas.append(coords)
        self.history_dates.append(date)
        self.hist_size+=1

    def detect_anomalies(self, img):
        simil_accumul = np.zeros(img.shape, dtype=np.float64)
        for i in range(self.hist_size):
            im = self.history_imgs[i]
            _, similarity = structural_similarity(im, img, full=True, channel_axis = 2)
            simil_accumul += similarity
        max_simil = np.max(simil_accumul)
        highlights = np.zeros(img.shape[:2], dtype=np.uint8)
        rindex = 0
        for row in simil_accumul:
            cindex = 0
            for pixel in row:
                p = pixel.copy()
                pixdiffs = (p - 0.2*max_simil > 0)
                if pixdiffs.any():
                    highlights[rindex][cindex] = 255
                cindex+=1
            rindex+=1

        contours, _ = cv2.findContours(highlights, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        count = len(contours)
        print(count)
        bounds = []
        for c in contours:
            co = c.copy()
            co = np.reshape(co, [s for s in c.shape if s != 1])
            if len(co.shape) == 1:
                co = np.reshape(co, (1, co.shape[0]))
            x, y = [p[0] for p in co], [p[1] for p in co]
            bounds.append([min(x), min(y), max(x), max(y)])
        return bounds

