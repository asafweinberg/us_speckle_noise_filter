from denoise import denoise_img
import numpy as np
import cv2
import math
import pickle
import os
from skimage.util import random_noise
from denoise import denoise_img
from enums import *
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from os import listdir
from os.path import isfile, join

mean = 0
noise_variance = 0.04
images_path="./metrics/images/US_images"
results_path="./metrics/output"
#results_path=".\\metrics\\output"
metrics_path="./metrics/results_metrics_file.txt"
#metrics_path=".\\metrics\\results_metrics_file.txt"

def run_metrics(laplacian_filter,number_layers):
    f = open(metrics_path,'w')
    only_files = [f for f in listdir(images_path) if isfile(join(images_path, f))]
    images_names=[f for f in only_files if ".png" in f]
    average_results={
        'mse': 0,
        'signal2noise': 0,
        'psnr': 0
        }
    results_list=[]

    for img_name in images_names:
        img = cv2.imread(join(images_path, img_name),0).astype(np.float32) / 255.0
        results = run_metrics_on_img(img, laplacian_filter,number_layers, img_name)
        results_list.append(results)

    for metrica in average_results.keys():
        avg=float(sum(result[metrica] for result in results_list)) / len(results_list)
        average_results[metrica]=avg

    for result in results_list:
        print_results(result)
    
    print_results(average_results, avg=True)
    

    with open(os.path.join(results_path, 'metric_results.txt'), "w") as file:
        for key, value in average_results.items():
            file.write(f"{key}: {value}\n")

def run_metrics_on_img(img, laplacian_filter,number_layers, img_name):
    noisy_img = add_speckle_noise(img)
    # plt.imsave(f'.\\metrics\\images\\noisy_{img_name}', noisy_img, cmap='gray')
    noisy_img = np.expand_dims(noisy_img, 2)

    clean_image = denoise_img(noisy_img, laplacian_filter, number_layers, pyr_method =PyrMethod.CV2 ,edge_filter=EdgeFilter.SOBEL_CV2,file_name=None, log=True)
    # plt.imsave(f'.\\metrics\\images\\clean_{img_name}', clean_image, cmap='gray')
    save_image_results(noisy_img, clean_image, f'{results_path}\\final_pair_{img_name}')
                       
    return({
        'mse': meansquareerror(img,clean_image),
        'signal2noise': signaltonoise(clean_image),
        'psnr': psnr(img,clean_image),
        'ssim': ssim(img,clean_image,data_range=clean_image.max() - clean_image.min())
    })


def add_speckle_noise(img):
    return random_noise(img, mode='speckle',var=noise_variance)



def meansquareerror(src, dst):
    if src.ndim == 3:
        src = src.mean(2)
        #dst = dst.ndim(2)
    mse = np.mean((src - dst) ** 2)
    return mse

def signaltonoise(src):
    a = np.asanyarray(src)
    m = np.mean(a)
    sd = np.std(a)
    return abs(10 * math.log10(math.pow(m,2) / math.pow(sd,2)))

def psnr(src, dst):
    if src.ndim == 3:
        src = src.mean(2)
        #dst = dst.mean(2)
    mse = np.mean((src - dst) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX =255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# def calc_ssim(src, dst):
#     return ssim(src, dst)


def print_results(metrics, avg=False):
    if avg: print('Average: ')
    print(f'MSE: {metrics["mse"]},    signal2noise: {metrics["signal2noise"]},    PSNR: {metrics["psnr"]}')


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
    run_metrics(laplacian,number_layers) 