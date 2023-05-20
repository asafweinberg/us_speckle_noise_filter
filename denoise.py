import numpy as np
import cv2
import matlab.engine
from scipy import ndimage
from generate_blurred_pyramid import *
from matlab_hadler import *
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from enums import *
from skimage import io, img_as_float, restoration
import skimage

from poisson_solver import poisson_solver_py

eng = matlab.engine.start_matlab()


#input image in grey scale and type float_32
def denoise_img(image, laplacian_filter, pyr_levels, pyr_method, edge_filter,preprocess_filter = Filters.NONE, postprocess_filter = Filters.NONE, file_name=None, log=False, range_correction=Range.HIST_MATCH):
    if max(image.shape)>1024: 
        scale_factor = 0.5
        scaled_shape_int=(int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))
        image=cv2.resize(image, scaled_shape_int)
        image = np.expand_dims(image, 2)

    original_image=image.squeeze()

    if preprocess_filter is not Filters.NONE:
        image = filter_image(original_image, preprocess_filter)
        image = np.expand_dims(image, 2)
    
    if log: print('creating gaussian pyramid')

    BlurredPyramid, padR, padC = create_pyramid(image, pyr_levels, pyr_method)
    W= np.abs(ndimage.convolve(np.abs(BlurredPyramid[pyr_levels-1]), laplacian_filter, mode='nearest'))

    if (W.max() > 0):
        W = W/ W.max()

    if log: print('finding maximum edges from pyramid layers')

    for i in range(pyr_levels-2,-1,-1):
        Gn = np.abs(ndimage.convolve(np.abs(BlurredPyramid[i]), laplacian_filter, mode='nearest'))
        W_expanded = pyramid_up(W,pyr_method)
        W = np.maximum(W_expanded, Gn)

        # TODO: remove the 0 padding, hurts this condition
        if (W.max() > 0):
            W = W / W.max()
    
    # if log: save_results(scaled_img, W, 'W')

    if log: print('edge detection')

    Gx, Gy = detect_edges(BlurredPyramid[0], edge_filter) #TODO: contrast stertching\ threshold

    if log: print('starting poisson solver')

    Wx = (0.5 * (W) + 1.0) * Gx
    Wy = (0.5 * (W) + 1.0) * Gy
    # if log: save_results(scaled_img, Wx, 'Wx_canny')
    # if log: save_results(scaled_img, Wy, 'Wy2_canny')


    diffused_img_sobel = poisson_solver_py(Wx, Wy, BlurredPyramid[0])

    # diffused_img_sobel = to_python(eng.poisson_solver_function(to_matlab(Wx, expand_dims = False),
    #                                                 to_matlab(Wy, expand_dims = False),
    #                                                 to_matlab(BlurredPyramid[0], expand_dims = False)), expand_dims = False)
    
    
    normalized = diffused_img_sobel / diffused_img_sobel.max()
    cropped = normalized[:-padR, :-padC]

    if postprocess_filter is not Filters.NONE:
        cropped = filter_image(cropped, postprocess_filter)
    

    img_float = correct_range(cropped, original_image, range_correction)
    


    if file_name:
        save_results(original_image, img_float, file_name)

    return img_float

def create_pyramid(image, number_of_layers, method):
    if method == PyrMethod.MATLAB:
        BlurredPyramid, padR, padC = generate_blurred_pyramid_func(eng, image, number_of_layers)
        return BlurredPyramid, padR, padC
    elif method == PyrMethod.CV2:
        BlurredPyramid, padR, padC = generate_blurred_pyramid_cv2_func(image, number_of_layers)
        return BlurredPyramid, padR, padC

def pyramid_up(W, method):
    if method == PyrMethod.MATLAB:
        return to_python(eng.my_impyramid(to_matlab(W, expand_dims=True), 'expand'), expand_dims=False)
    if method == PyrMethod.CV2:
        return cv2.pyrUp(W)

def detect_edges(image, edge_filter=EdgeFilter.SOBEL_ND_IMAGE):
    if edge_filter == EdgeFilter.SOBEL_ND_IMAGE:
        Gy = ndimage.sobel(image,axis=0,mode='constant')
        Gx = ndimage.sobel(image,axis=1,mode='constant')
    elif edge_filter == EdgeFilter.SOBEL_CV2:
        Gx = cv2.Sobel(image, cv2.CV_64F, 1, 0)
        Gy = cv2.Sobel(image, cv2.CV_64F, 0, 1)
    elif edge_filter == EdgeFilter.SCHARR:
        Gx = cv2.Scharr(image, cv2.CV_64F, 1, 0)
        Gy = cv2.Scharr(image, cv2.CV_64F, 0, 1) 
    elif edge_filter == EdgeFilter.CANNY:
        edge = cv2.Canny(img, 0.2, 0.7, apertureSize=5, L2gradient =True)
        Gx = cv2.Sobel(edge, cv2.CV_64F, 1, 0)
        Gy = cv2.Sobel(edge, cv2.CV_64F, 0, 1)
    return Gx,Gy

def contrast_strech_transform(image, f1=0.2, f2=0.8, alpha=0.5, beta=1.3333, gamma=0.5, g1=0.1, g2=0.9):
    low_range=np.where(image<f1)
    middle_range=np.where(np.logical_and(image>=f1 , image<=f2))
    high_range=np.where(image>f2)

    image[low_range]=alpha*image[low_range]
    #image[image<f1]*=beta

    image[middle_range]=beta*(image[middle_range]-f1)+g1
    image[high_range]=gamma*(image[high_range]-f2)+g2



def filter_image(image, filter_type):
    if filter_type == Filters.NLM:
        s = image.squeeze()
        filtered_image = restoration.denoise_nl_means(s, h=0.01, patch_size=5, fast_mode=True)
        filtered_image = np.expand_dims(filtered_image, 2)
    
    if filter_type == Filters.BILATERAL:
        image = image.astype("float32")
        image = cv2.bilateralFilter(image, 5, 2, 2)
    if filter_type == Filters.KUAN:
        image =  restoration.denoise_tv_chambolle(image, weight=0.05)


    return image

def correct_range(image, original_image, range_correction):
    if range_correction==Range.HIST_MATCH:
        return skimage.exposure.match_histograms(image, original_image)

    if range_correction==Range.NORMALIZE:
        min_val=np.min(image)
        max_val=np.max(image)
        img_float=(image-min_val)/(max_val-min_val)
        return img_float
    
    if range_correction==Range.CONTRAST_STRETCH:
        min_val=np.min(image)
        max_val=np.max(image)
        img_float=(image-min_val)/(max_val-min_val)
        contrast_strech_transform(img_float)
        return img_float


def save_results(origin, denoised, file_name):
    _, axarr = plt.subplots(ncols=2, figsize=(14,14))
    axarr[0].imshow(origin, cmap='gray')
    axarr[0].axis('off')
    axarr[1].imshow(denoised, cmap='gray')
    axarr[1].axis('off')
    plt.savefig(f'./results/{file_name}.png', bbox_inches='tight')


if __name__ == "__main__":
    file_name = 'UStest.png'
    N = 4
    file_name_extension = f'poisson_python'
    save_name = f'{file_name[:-4]}_{file_name_extension}'

    img = cv2.imread(f'./data/{file_name}',0).astype(np.float32)/255.0
    laplacian = np.array([0, -1, 0, -1, 4, -1, 0, -1, 0]).reshape((3, 3))

    img = np.expand_dims(img, 2) #adds another dim

   
    MATLAB_img =denoise_img(img, laplacian, pyr_levels=N, pyr_method=PyrMethod.CV2, edge_filter=EdgeFilter.SOBEL_CV2,file_name=save_name,log=True,range_correction=Range.CONTRAST_STRETCH)


