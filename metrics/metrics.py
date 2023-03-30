import numpy as np
import cv2
import math
from skimage.util import random_noise
from denoise import denoise_img

mean = 0
noise_variance = 0.5 

def run_metrics(laplacian_filter):
    img = cv2.imread(".\\metrics\\images\\lena.png",0) / 255.0
    img = np.expand_dims(img, 2)
    results = run_metrics_on_img(img, laplacian_filter)

    print_results(results)

def run_metrics_on_img(img, laplacian_filter):
    noisy_img = add_speckle_noise(img)
    clean_image = denoise_img(img, laplacian_filter)

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
    print(f'MSE: {metrics["mse"]},    signal2noise: {metrics["signal2noise"]},    PSNR: {metrics["psnr"]}')
