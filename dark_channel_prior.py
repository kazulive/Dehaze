import cv2
import numpy as np

class DarkChannelPrior:

    def __init__(self,
                 wsize,
                 ratio):
        """
        :param wsize : size of window for dark channel
        :param ratio: ratio of pixels used to estimate the atmosphere
        """
        self.wsize = wsize
        self.ratio = ratio


    def dark_channel(self, img):
        b, g, r = cv2.split(img)
        min_img = cv2.min(r, cv2.min(g, b))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.wsize, self.wsize))
        dark = cv2.erode(min_img, kernel)
        return dark

    def get_atmosphere(self, img, dark):
        img = img.reshape([-1, 3])
        img_dark = dark.flatten()
        top_k_index = np.argsort(img_dark)[-int(img_dark.size * self.ratio):]
        return np.max(np.take(img, top_k_index, axis=0), axis=0)

    def get_transmission(self, img, A):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        return 1.0 - cv2.morphologyEx(self.dark_channel(img/A), cv2.MORPH_OPEN, kernel)

    def dehaze(self, img):
        dark = self.dark_channel(img)
        A = self.get_atmosphere(img, dark)
        print(dark.shape, A.shape)
        return self.get_transmission(img, A)
