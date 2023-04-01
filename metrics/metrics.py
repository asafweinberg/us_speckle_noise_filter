import numpy as np
import cv2
import math
from skimage.util import random_noise
from denoise import denoise_img
from enums import *
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join

mean = 0
noise_variance = 0.04
images_path=".\\metrics\\images"

def run_metrics(laplacian_filter,number_layers):
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
        results = run_metrics_on_img(img, laplacian_filter,number_layers)
        results_list.append(results)

    for metrica in average_results.keys():
        avg=float(sum(result[metrica] for result in results_list)) / len(results_list)
        average_results[metrica]=avg
    
    print_results(average_results)

def run_metrics_on_img(img, laplacian_filter,number_layers):
    noisy_img = add_speckle_noise(img)
    #plt.imsave(f'.\\metrics\\images\\noisy2.png', noisy_img, cmap='gray')
    noisy_img = np.expand_dims(noisy_img, 2)

    clean_image = denoise_img(noisy_img, laplacian_filter, number_layers, PyrMethod.CV2, edge_filter=EdgeFilter.SOBEL_CV2,file_name="LenaCV2")
    #plt.imsave(f'.\\metrics\\images\\clean.png', clean_image, cmap='gray')

    return({
        'mse': meansquareerror(img,clean_image),
        'signal2noise': signaltonoise(clean_image),
        'psnr': psnr(img,clean_image)
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


def print_results(metrics):
    print(f'Average: MSE: {metrics["mse"]},    signal2noise: {metrics["signal2noise"]},    PSNR: {metrics["psnr"]}')

