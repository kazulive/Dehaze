import cv2
import numpy as np

import dark_channel_prior as DCP
import padding as PD

class VariationalModel(object):

    def __init__(self, t0, alpha, beta, lam1, lam2):
        self.t0 = t0
        self.alpha = alpha
        self.beta = beta
        self.lam1 = lam1
        self.lam2 = lam2
        self.kernel = np.array([[0, 0, 0],
                                [-1, 0, 1],
                                [0, 0, 0]])

    def shrinkage(self, img, bh, bv, lam):
       former_h = cv2.filter2D(img.astype(dtype=np.float32), cv2.CV_32F, self.kernel) + bh
       former_v = cv2.filter2D(img.astype(dtype=np.float32), cv2.CV_32F, self.kernel.T) + bv
       latter = 1. / (2. * lam)

       tmp_h = np.maximum(np.abs(former_h) - latter, 0.0)
       tmp_v = np.maximum(np.abs(former_v) - latter, 0.0)

       dh = np.sign(former_h) * tmp_h
       dv = np.sign(former_v) * tmp_v

       return dh, dv

    def diff(self, img, bh, bv, dh, dv):
        new_bh = bh + cv2.filter2D(img.astype(dtype=np.float32), cv2.CV_32F, self.kernel) - dh
        new_bv = bv + cv2.filter2D(img.astype(dtype=np.float32), cv2.CV_32F, self.kernel.T) - dv
        return new_bh, new_bv

    def optimization(self, img):
        # 配列用意
        H, W = img.shape
        R = np.zeros((H, W), dtype=np.float32)                                                                      # 出力画像              => (W, H, 1) float32型
        T = self.t0                                                                                                 # 透過率マップ          => (W, H, 1) float32型

        bh = np.zeros((H, W), dtype=np.float32)
        bv = np.zeros((H, W), dtype=np.float32)
        vh = np.zeros((H, W), dtype=np.float32)
        vv = np.zeros((H, W), dtype=np.float32)

        F_conj_h = PD.psf2otf(np.expand_dims(np.array([1, -1]), axis=1), img.shape[:2]).conjugate()                 # FFT derivative operateor horizontal
        F_conj_v = PD.psf2otf(np.expand_dims(np.array([1, -1]), axis=1).T, img.shape[:2]).conjugate()               # FFT derivative operateor verical
        F_delta, F_div = PD.getKernel(img)

        count = 0
        while(count < 5):
            # Step1.
            dh, dv = self.shrinkage(R, bh, bv, self.lam1)
            phi1 = F_conj_h * np.fft.fft2(dh - bh) + F_conj_v * np.fft.fft2(dv - bv)
            # Step2.
            tmp = img / np.maximum(T, 0.3)
            top = np.fft.fft2(tmp) + self.alpha * self.lam1 * phi1
            bottom = F_delta + self.alpha * self.lam1 * F_div
            R = np.real(np.fft.ifft2(top / bottom))
            # Step3.
            bh, bv = self.diff(R, bh, bv, dh, dv)
            R = np.maximum(R, img)
            # Step4.
            uh, uv = self.shrinkage(T, vh, vv, self.lam2)
            phi2 = F_conj_h * np.fft.fft2(uh - vh) + F_conj_v * np.fft.fft2(uv - vv)
            top = np.fft.fft2(img/(R + 0.001)) + self.beta * self.lam2 * phi2
            bottom = F_delta + self.beta * self.lam2 * F_div
            T = np.real(np.fft.ifft2(top / bottom))
            # Step5.
            vh, vv = self.diff(T, vh, vv, uh, uv)
            T = np.maximum(np.minimum(T, 1.0), 0.0)
            count += 1
            #cv2.imshow("result", R.astype(dtype=np.uint8))
            cv2.imshow("T", T)
            cv2.waitKey(0)
        return R, T

if __name__ == '__main__':
    img = cv2.imread("09.bmp")
    img = img.astype(dtype=np.float32)
    A, inital_transmission = DCP.DarkChannelPrior(wsize=15, ratio=0.001).dehaze((img).astype(dtype=np.float32))
    b, g, r = cv2.split(img)
    b = 1.0 - b / A[0]
    g = 1.0 - g / A[1]
    r = 1.0 - r / A[2]
    b_output, b_trans_map = VariationalModel(inital_transmission, 0.1, 0.1, 0.1, 10).optimization(b)
    g_output, g_trans_map = VariationalModel(inital_transmission, 0.1, 0.1, 0.1, 10).optimization(g)
    r_output, r_trans_map = VariationalModel(inital_transmission, 0.1, 0.1, 0.1, 10).optimization(r)
    b_output = A[0] * (1. - b_output)
    g_output = A[1] * (1. - g_output)
    r_output = A[2] * (1. - r_output)
    output = cv2.merge((b_output, g_output, r_output))
    output = np.maximum(np.minimum(output, 255.), 0.0)
    cv2.imshow("result", output.astype(dtype=np.uint8))
    cv2.imshow("trans", ((b_trans_map + g_trans_map + r_trans_map)/3.0))
    cv2.imshow("init", inital_transmission)
    #cv2.imshow("b_result", b_trans_map)
    #cv2.imshow("g_result", g_trans_map)
    #cv2.imshow("r_result", r_trans_map)
    cv2.waitKey(0)



