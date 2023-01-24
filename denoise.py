import numpy as np
import cv2
import matlab.engine
from scipy import ndimage
from generate_blurred_pyramid import generate_blurred_pyramid_func
#input image in grey scale and type float_32
def denoise_img(image):
    eng = matlab.engine.start_matlab()
    scale_factor = 0.75
    N = 3
    L = 0.5 * np.array([0, -1, 0, -1, 4, -1, 0, -1, 0]).reshape((3, 3))

    scaled_shape_int=(int(image.shape[0] * scale_factor),  int(image.shape[1] * scale_factor))

    #line 9 in matlab
    scaled_img = cv2.resize(image, scaled_shape_int).reshape(*scaled_shape_int,image.shape[2])
    
    BlurredPyramid, padR, padC=generate_blurred_pyramid_func(eng,scaled_img,N)
    # if BlurredPyramid.ndim == 2:
    #     BlurredPyramid=np.expand_dims(BlurredPyramid,0)
    W= np.abs(ndimage.convolve(np.abs(BlurredPyramid[N-1]),L,mode='nearest')) #TODO: NOT TESTED

    if (W.max() > 0):
        W = W/ W.max()

    for i in range(N-2,-1,-1):
        Gn=np.abs(ndimage.convolve(np.abs(BlurredPyramid[i]),L,mode='nearest'))
        W_expanded=eng.my_impyramid(W, 'expand')
        W = np.max(W_expanded, Gn)

        if (W.max() > 0):
            W = W / W.max()





img=np.random.rand(256,256,1)
denoise_img(img)