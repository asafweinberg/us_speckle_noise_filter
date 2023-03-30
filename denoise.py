import numpy as np
import cv2
import matlab.engine
from scipy import ndimage
from generate_blurred_pyramid import generate_blurred_pyramid_func
from matlab_hadler import *
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from enums import *

eng = matlab.engine.start_matlab()


#input image in grey scale and type float_32
def denoise_img(image, file_name, laplacian_filter, pyr_method=PyrMethod.MATLAB, edge_filter=EdgeFilter.SOBEL_ND_IMAGE):
    scale_factor = 0.7
    N = 3

    scaled_shape_int=(int(image.shape[0] * scale_factor),  int(image.shape[1] * scale_factor))

    scaled_img = image
    
    BlurredPyramid, padR, padC = create_pyramid(scaled_img, N, pyr_method)
    W= np.abs(ndimage.convolve(np.abs(BlurredPyramid[N-1]), laplacian_filter, mode='nearest'))

    if (W.max() > 0):
        W = W/ W.max()

    for i in range(N-2,-1,-1):
        Gn = np.abs(ndimage.convolve(np.abs(BlurredPyramid[i]), laplacian_filter, mode='nearest'))
        W_expanded = pyramid_up(W)
        W = np.maximum(W_expanded, Gn)

        if (W.max() > 0):
            W = W / W.max()

    # ----------------------------------cv2 Sobel--------------------------------------------------------

    # kernel_x = np.array([1,0,-1,2,0,-2,1,0,-1]).reshape((3, 3))
    # kernel_y = np.array([1,2,1,0,0,0,-1,-2,-1]).reshape((3, 3))

    # # Gx = ndimage.convolve(np.abs(BlurredPyramid[0]), kernel_x, mode='nearest')
    # # Gy = ndimage.convolve(np.abs(BlurredPyramid[0]), kernel_y, mode='nearest')

    # diffused_img = to_python(eng.poisson_solver_function(to_matlab((0.5 * (W) + 1.0) * Gx, expand_dims = False),
    #                                                      to_matlab((0.5 * (W) + 1.0) * Gy, expand_dims = False),
    #                                                      to_matlab(BlurredPyramid[0], expand_dims = False)), expand_dims = False)

    # # Rrgb = ConvertFormOpponentToRgb1( R );

    # normalized = diffused_img / diffused_img.max()
    # cropped = normalized[:-padR, :-padC]
    # clipped = np.clip(cropped, 0, 1)
    # plt.imsave(f'./results/{file_name}_cv2_sobel.png', clipped, cmap='gray')


    # ----------------------------------ndimage Sobel--------------------------------------------------------
    
    Gx, Gy = detect_edges(BlurredPyramid[0], edge_filter)

    diffused_img_sobel = to_python(eng.poisson_solver_function(to_matlab((0.5 * (W) + 1.0) * Gx, expand_dims = False),
                                                    to_matlab((0.5 * (W) + 1.0) * Gy, expand_dims = False),
                                                    to_matlab(BlurredPyramid[0], expand_dims = False)), expand_dims = False)
    normalized = diffused_img_sobel / diffused_img_sobel.max()
    cropped = normalized[:-padR, :-padC]
    clipped = cropped
    # clipped = np.clip(cropped, 0, 1)
    plt.imsave(f'./results/{file_name}_ndimage_sobel_no_clip.png', clipped, cmap='gray')

    return normalized

def create_pyramid(image, number_of_layers, method=PyrMethod.MATLAB):
    if method == PyrMethod.MATLAB:
        BlurredPyramid, padR, padC = generate_blurred_pyramid_func(eng, image, number_of_layers)
        return BlurredPyramid, padR, padC
    elif method == PyrMethod.CV2:
        return

def pyramid_up(W, method=PyrMethod.MATLAB):
    if method == PyrMethod.MATLAB:
        return to_python(eng.my_impyramid(to_matlab(W, expand_dims=True), 'expand'), expand_dims=False)
    if method == PyrMethod.CV2:
        return

def detect_edges(image, edge_filter=EdgeFilter.SOBEL_ND_IMAGE):
    if EdgeFilter.SOBEL_ND_IMAGE:
        Gy = ndimage.sobel(image,axis=0,mode='constant')
        Gx = ndimage.sobel(image,axis=1,mode='constant')
    elif EdgeFilter.SOBEL_CV2:
        Gx = cv2.Sobel(image, cv2.CV_64F, 1, 0)
        Gy = cv2.Sobel(image, cv2.CV_64F, 0, 1)
    return Gx,Gy


file_name = 'speckle_noise_example.ppm'
img = cv2.imread(f'./data/{file_name}',0).astype(np.float32)/255.0
laplacian = 0.5 * np.array([0, -1, 0, -1, 4, -1, 0, -1, 0]).reshape((3, 3))

img = np.expand_dims(img, 2)
# img = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 2)
# img=np.random.rand(256,256,1)
denoise_img(img, file_name, laplacian, pyr_method=PyrMethod.MATLAB, edge_filter=EdgeFilter.SOBEL_CV2)

