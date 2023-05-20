from denoise import denoise_img
import numpy as np
import cv2
import math
import csv
import os
from skimage import restoration
from skimage.util import random_noise
from denoise import denoise_img
from enums import *
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import datetime

laplacian = np.array([0, -1, 0, -1, 4, -1, 0, -1, 0]).reshape((3, 3))
mean = 0
noise_variance = 0.04
general_images_path="./metrics/images/general_images"
us_images_path="./metrics/images/US_images"
results_path="./metrics/output"

now = datetime.datetime.now()
time = now.strftime("%m_%d_%H_%M_%S")

def run_other_methods():
    rows = []
    all_keys = None
    for method in Methods:
        print(f'running method {method}')
        average_results = run_by_method(method, False)
        if all_keys == None:
            all_keys = average_results.keys()
            header = ['method', 'dataset'] + list(sorted(all_keys))
            rows = [header]
        row_gen = [method, 'general'] + [format(average_results.get(key, ''), '.5f') for key in sorted(all_keys)]
        
        # average_results = run_by_method(method, True)
        # row_us = [method, 'US'] + [format(average_results.get(key, ''), '.5f') for key in sorted(all_keys)]
        rows += [row_gen]
        # rows += [row_us]
    
    with open(os.path.join(results_path, f'metric_results_other_methods.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)



def run_by_method(method, run_on_us_images):
    images_path = us_images_path if run_on_us_images else general_images_path
    only_files = [f for f in listdir(images_path) if isfile(join(images_path, f))]
    images_names=[f for f in only_files if ".png" in f]

    average_results={
            'mse': 0,
            'signal2noise': 0,
            'psnr': 0,
            'ssim':0
            }
    results_list=[]    
    for img_name in tqdm(images_names):
        image = cv2.imread(join(images_path, img_name),0).astype(np.float32) / 255.0
        noisy_img = add_speckle_noise(image).astype(np.float32)

        if method == Methods.MEDIAN:
            filtered = apply_filter(noisy_img, cv2.medianBlur, ksize=5)
        elif method == Methods.GAUSSIAN:
            filtered = apply_filter(noisy_img, cv2.GaussianBlur, ksize=(5, 5), sigmaX=0)
        elif method == Methods.NLM:
            filtered = apply_filter(noisy_img, restoration.denoise_nl_means, h=0.01, patch_size=5, fast_mode=True)
        elif method == Methods.BILATERAL:
            filtered= apply_filter(noisy_img, cv2.bilateralFilter, 5, 2, 2)
        elif method == Methods.SRAD:
            filtered = apply_filter(noisy_img, restoration.denoise_tv_bregman, weight=0.1, max_num_iter=10, eps=0.1, isotropic=False)
        elif method == Methods.KUAN:
            filtered= apply_filter(noisy_img, restoration.denoise_tv_chambolle, weight=0.1)
        elif method == Methods.OURS: 
            noisy_img = np.expand_dims(noisy_img, 2)
            filtered = denoise_img(noisy_img, 
                                   laplacian_filter=laplacian, 
                                   pyr_levels=4, 
                                   pyr_method=PyrMethod.CV2, 
                                   edge_filter=EdgeFilter.SCHARR, 
                                   preprocess_filter= Filters.BILATERAL, 
                                   postprocess_filter=Filters.NONE, 
                                   range_correction=Range.CONTRAST_STRETCH, 
                                   log=False)

        if 'lena' in img_name:
            save_metric_image_results(image, noisy_img, filtered, os.path.join(results_path, f'lena_{method.name}.png'))

        results = compute_all_metrics(image,filtered)
        results_list.append(results)

    for metrica in average_results.keys():
        avg=float(sum(result[metrica] for result in results_list)) / len(results_list)
        average_results[metrica]=avg
    return average_results


def compare_visually():
    images_path = us_images_path
    only_files = [f for f in listdir(images_path) if isfile(join(images_path, f))]
    images_names=[f for f in only_files if ".png" in f]

    our_edge_filter=EdgeFilter.SCHARR
    our_preprocess_filter= Filters.KUAN
    our_range_correction=Range.CONTRAST_STRETCH


    for img_name in tqdm(images_names):
        image = cv2.imread(join(images_path, img_name),0).astype(np.float32) / 255.0
        reslts = [('origin', image)]
        for method in Methods:
            if method == Methods.MEDIAN:
                filtered = apply_filter(image, cv2.medianBlur, ksize=5)
            elif method == Methods.GAUSSIAN:
                filtered = apply_filter(image, cv2.GaussianBlur, ksize=(5, 5), sigmaX=0)
            elif method == Methods.NLM:
                filtered = apply_filter(image, restoration.denoise_nl_means, h=0.01, patch_size=5, fast_mode=True)
            elif method == Methods.BILATERAL:
                filtered= apply_filter(image, cv2.bilateralFilter, 5, 2, 2)
            elif method == Methods.SRAD:
                filtered = apply_filter(image, restoration.denoise_tv_bregman, weight=0.1, max_num_iter=10, eps=0.1, isotropic=False)
            elif method == Methods.KUAN:
                filtered= apply_filter(image, restoration.denoise_tv_chambolle, weight=0.1)
            elif method == Methods.OURS: 
                image = np.expand_dims(image, 2)
                filtered = denoise_img(image, 
                                        laplacian_filter=laplacian, 
                                        pyr_levels=4, 
                                        pyr_method=PyrMethod.CV2, 
                                        edge_filter=our_edge_filter,
                                        preprocess_filter=our_preprocess_filter,
                                        postprocess_filter=Filters.NONE, 
                                        range_correction=our_range_correction, 
                                        log=False)
                
            reslts.append((method.name, filtered))
        
        save_multi_method(img_name, reslts, our_edge_filter, our_preprocess_filter, our_range_correction)


def save_multi_method(image_name, images, edge_filter, preprocess_filter, range_correction):
    exp_name = f'{edge_filter.name}_{preprocess_filter.name}_{range_correction.name}'
    fig, axs = plt.subplots(3, 3, figsize=(10,10))

    for i, ax in enumerate(axs.flat):
        if i < len(images):
            ax.imshow(images[i][1], cmap='gray')
            ax.set_title(images[i][0], fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.label_outer()
    
    plt.tight_layout()
    dir = f'{results_path}\\experiments_{exp_name}_{time}'
    if not os.path.exists(dir):
        os.makedirs(dir)    
    plt.savefig(f'{dir}\\{image_name}')
      



def apply_filter(image, filter_func, *args, **kwargs):
    filtered_image = filter_func(image, *args, **kwargs)
    return filtered_image


def run_metrics(laplacian_filter,
                number_layers, 
                edge_filter, 
                preprocess_filter = Filters.NONE, 
                postprocess_filter = Filters.NONE,
                range_correction = Range.HIST_MATCH,
                run_on_us_images=True,
                log_results=True):
    images_path = us_images_path if run_on_us_images else general_images_path
    only_files = [f for f in listdir(images_path) if isfile(join(images_path, f))]
    images_names=[f for f in only_files if ".png" in f]
    average_results={
        'mse': 0,
        'signal2noise': 0,
        'psnr': 0,
        'ssim':0
        }
    results_list=[]

    for img_name in tqdm(images_names):
        img = cv2.imread(join(images_path, img_name),0).astype(np.float32) / 255.0
        results = run_metrics_on_img(img,
                                     laplacian_filter,
                                     number_layers, 
                                     edge_filter, 
                                     preprocess_filter, 
                                     postprocess_filter,
                                     range_correction, 
                                     img_name)
        results_list.append(results)

    for metrica in average_results.keys():
        avg=float(sum(result[metrica] for result in results_list)) / len(results_list)
        average_results[metrica]=avg

    if log_results:
        for result in results_list:
            print_results(result)
        print_results(average_results, avg=True)
        with open(os.path.join(results_path, 'metric_results.txt'), "w") as file:
            for key, value in average_results.items():
                file.write(f"{key}: {value}\n")
        return average_results
    else:
        return average_results

def run_metrics_on_img(img, 
                       laplacian_filter,
                       number_layers,
                       edge_filter, 
                       preprocess_filter = Filters.NONE, 
                       postprocess_filter = Filters.NONE,
                       range_correction = Range.HIST_MATCH, 
                       img_name=None):
    noisy_img = add_speckle_noise(img)
    # plt.imsave(f'.\\metrics\\images\\noisy_{img_name}', noisy_img, cmap='gray')
    noisy_img = np.expand_dims(noisy_img, 2)

    clean_image = denoise_img(noisy_img, laplacian_filter, number_layers, PyrMethod.CV2,
                              edge_filter=edge_filter,
                              preprocess_filter=preprocess_filter,
                              postprocess_filter = postprocess_filter,
                              range_correction = range_correction,
                              log=False)
    # plt.imsave(f'.\\metrics\\images\\clean_{img_name}', clean_image, cmap='gray')
    save_image_results(noisy_img, clean_image, f'{results_path}\\final_pair_{img_name}')
                       
    return compute_all_metrics(img,clean_image)


def add_speckle_noise(img):
    return random_noise(img, mode='speckle',var=noise_variance)


def compute_all_metrics(img,clean_image):
    return {
        'mse': meansquareerror(img,clean_image),
        'signal2noise': signaltonoise(clean_image),
        'psnr': psnr(img,clean_image),
        'ssim': ssim(img,clean_image,data_range=clean_image.max() - clean_image.min())
    }


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


def print_results(metrics, avg=False, log_results=True):
    results_str = f'MSE: {metrics["mse"]},    signal2noise: {metrics["signal2noise"]},    PSNR: {metrics["psnr"]},     SSIM: {metrics["ssim"]}'
    if log_results:
        if avg: print('Average:')
        print(results_str)
    else:
        return results_str


def save_image_results(origin, denoised, file_name):
    _, axarr = plt.subplots(ncols=2, figsize=(14,14))
    axarr[0].imshow(origin, cmap='gray')
    axarr[0].axis('off')
    axarr[1].imshow(denoised, cmap='gray')
    axarr[1].axis('off')
    plt.savefig(file_name, bbox_inches='tight')


def save_metric_image_results(origin, noisy, denoised, file_name):
    _, axarr = plt.subplots(ncols=3, figsize=(14,14))
    axarr[0].imshow(origin, cmap='gray')
    axarr[0].axis('off')
    axarr[1].imshow(noisy, cmap='gray')
    axarr[1].axis('off')
    axarr[2].imshow(denoised, cmap='gray')
    axarr[2].axis('off')
    plt.savefig(file_name, bbox_inches='tight')


if __name__ == "__main__":
    # laplacian = np.array([0, -1, 0, -1, 4, -1, 0, -1, 0]).reshape((3, 3))
    # number_layers=4
    # run_metrics(laplacian,number_layers) 
    # run_other_methods()
    compare_visually()