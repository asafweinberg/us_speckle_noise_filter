import numpy as np
import cv2
import matlab.engine
from scipy import ndimage

#input image in grey scale and type float_32
def denoise_img(image):
    eng = matlab.engine.start_matlab()
    scale_factor = 1
    N = 3
    L = 0.5 * np.array([0, -1, 0, -1, 4, -1, 0, -1, 0]).reshape((3, 3))

    #line 9 in matlab
    scaled_img = cv2.resize(image, (image.shape[0] * scale_factor, image.shape[1] * scale_factor))
    BlurredPyramid, padR, padC=eng.GenerateBlurredPyramid(image,N-1)
    W= np.abs(ndimage.convolve(np.abs(BlurredPyramid[N-1]),L,mode='nearest'))

    if (W.max() > 0):
        W = W/ W.max()

    for i in range(N-2,-1,-1):
        Gn=np.abs(ndimage.convolve(np.abs(BlurredPyramid[i]),L,mode='nearest'))
        W = np.max(eng.my_impyramid(W, 'expand'), Gn)

        if (W.max() > 0):
            W = W / W.max()





img=np.random.rand(256,256)
denoise_img(img)