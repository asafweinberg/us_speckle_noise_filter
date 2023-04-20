import numpy as np
from metrics.metrics import run_metrics
import os
from os import listdir
from os.path import isfile, join
import cv2
from denoise import denoise_img
from enums import *
import matplotlib.pyplot as plt


def calc_metrics(laplacian_filter,number_layers):
    run_metrics(laplacian_filter,number_layers) 



def clean_images(laplacian_filter,number_layers):
    images_path=".\\test_images\\images"
    results_path=".\\test_images\\output"

    only_files = [f for f in listdir(images_path) if isfile(join(images_path, f))]
    images_names=[f for f in only_files if ".png" in f]

    for img_name in images_names:
        img = cv2.imread(join(images_path, img_name),0).astype(np.float32) / 255.0
        noisy_img = np.expand_dims(img, 2)
        clean_image = denoise_img(noisy_img, laplacian_filter, number_layers, PyrMethod.CV2, edge_filter=EdgeFilter.SOBEL_CV2,file_name='eq'+img_name, log=True)
        save_image_results(img, clean_image,  f'{results_path}\\clip_final_pair_{img_name}')



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
    # calc_metrics(laplacian,number_layers) 
    clean_images(laplacian,number_layers) 
