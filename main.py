import numpy as np
from metrics import run_metrics
import os
from os import listdir
from os.path import isfile, join
import cv2
from denoise import denoise_img
from enums import *
import matplotlib.pyplot as plt

images_path=".\\test_images\\images"
results_path=".\\test_images\\output"


def calc_metrics(laplacian_filter,number_layers):
    run_metrics(laplacian_filter,number_layers) 



def denoise_multiple_same_method(laplacian_filter,number_layers, edge_filter, preprocess_filter = Filters.NONE, postprocess_filter = Filters.NONE, postfix=''):
    only_files = [f for f in listdir(images_path) if isfile(join(images_path, f))]
    images_names = [f for f in only_files if ".png" in f]
    clean_images = {name: [] for name in images_names}

    for img_name in images_names:
        denoise_single_image(img_name, 
                             laplacian_filter,
                             number_layers, 
                             edge_filter, 
                             preprocess_filter, 
                             postprocess_filter, 
                             postfix)
        clean_images[img_name].append(())
        


def denoise_single_image(img_name,laplacian_filter, number_layers, edge_filter, preprocess_filter = Filters.NONE, postprocess_filter = Filters.NONE, postfix='')
    img = cv2.imread(join(images_path, img_name),0).astype(np.float32) / 255.0
    noisy_img = np.expand_dims(img, 2)
    clean_image = denoise_img(noisy_img, laplacian_filter, number_layers, PyrMethod.CV2, 
                                edge_filter=edge_filter,preprocess_filter=preprocess_filter,postprocess_filter = Filters.NONE, file_name=img_name, log=True)

    exp_path = f'{results_path}\\{exp_name}'
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    save_image_results(img, clean_image,  f'{exp_path}\\{img_name}')
    return clean_image

def get_experiment_name(edge_filter, number_layers, preprocess_filter, postprocess_filter):
    exp_name = f'{edge_filter.name}_{number_layers}_pre_{preprocess_filter.name}_post_{postprocess_filter.name}_{postfix}'


def save_image_results(origin, denoised, file_name):
    _, axarr = plt.subplots(ncols=2, figsize=(14,14))
    axarr[0].imshow(origin, cmap='gray')
    axarr[0].axis('off')
    axarr[1].imshow(denoised, cmap='gray')
    axarr[1].axis('off')
    plt.savefig(file_name, bbox_inches='tight')


if __name__ == "__main__":
    laplacian = np.array([0, -1, 0, -1, 4, -1, 0, -1, 0]).reshape((3, 3))
    number_layers=4
    edge_filter = EdgeFilter.SOBEL_CV2
    preprocess_filter = Filters.NLM
    postprocess_filter = Filters.NLM

    calc_metrics(laplacian,number_layers) 
    #clean_images(laplacian,number_layers, edge_filter, preprocess_filter=preprocess_filter) 
