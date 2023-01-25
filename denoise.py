import numpy as np
import cv2
import matlab.engine
from scipy import ndimage
from generate_blurred_pyramid import generate_blurred_pyramid_func
from matlab_hadler import *
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


#input image in grey scale and type float_32
def denoise_img(image):
    eng = matlab.engine.start_matlab()
    scale_factor = 0.7
    N = 3
    L = 0.5 * np.array([0, -1, 0, -1, 4, -1, 0, -1, 0]).reshape((3, 3))

    scaled_shape_int=(int(image.shape[0] * scale_factor),  int(image.shape[1] * scale_factor))

    #line 9 in matlab
    # scaled_img = cv2.resize(image, scaled_shape_int).reshape(*scaled_shape_int, 1)
    scaled_img = image
    
    BlurredPyramid, padR, padC = generate_blurred_pyramid_func(eng, scaled_img, N)
    # if BlurredPyramid.ndim == 2:
    #     BlurredPyramid=np.expand_dims(BlurredPyramid,0)
    W= np.abs(ndimage.convolve(np.abs(BlurredPyramid[N-1]), L, mode='nearest'))

    if (W.max() > 0):
        W = W/ W.max()

    for i in range(N-2,-1,-1):
        Gn = np.abs(ndimage.convolve(np.abs(BlurredPyramid[i]), L, mode='nearest'))
        W_expanded = to_python(eng.my_impyramid(to_matlab(W, expand_dims=True), 'expand'), expand_dims=False)
        W = np.maximum(W_expanded, Gn)

        if (W.max() > 0):
            W = W / W.max()
    
    # Gx = ndimage.sobel(BlurredPyramid[0],axis=0,mode='constant')
    # Gy = ndimage.sobel(BlurredPyramid[0],axis=1,mode='constant')

    Gx = cv2.Sobel(BlurredPyramid[0], cv2.CV_32F, 1, 0)
    Gy = cv2.Sobel(BlurredPyramid[0], cv2.CV_32F, 0, 1)

    diffused_img = to_python(eng.poisson_solver_function(to_matlab(1*(0.5*(W)+1.0)*Gx, expand_dims = False),
                                                         to_matlab(1*(0.5*(W)+1.0)*Gy, expand_dims = False),
                                                         to_matlab(BlurredPyramid[0], expand_dims = False)), expand_dims = False)

    # Rrgb = ConvertFormOpponentToRgb1( R );

    normalized = diffused_img / diffused_img.max()

    plt.imsave('./results/a.png',diffused_img, cmap='gray')
    return diffused_img



img = cv2.imread('./data/UStest.png',0).astype(np.float32)/255.0
img = np.expand_dims(img, 2)
# img = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 2)
# img=np.random.rand(256,256,1)
denoise_img(img)