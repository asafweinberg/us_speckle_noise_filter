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
from scipy.special import kl_div

laplacian = np.array([0, -1, 0, -1, 4, -1, 0, -1, 0]).reshape((3, 3))
mean = 0
noise_variance = 0.1
general_images_path="./metrics/images/general_images"
#us_images_path="./metrics/images/US_images"
us_images_path="test_images/images/very_speckle"
# us_images_path="test_images/images/very_speckle/metric_test"
# results_path="./metrics/output"
results_path="./metrics/output"

now = datetime.datetime.now()
time = now.strftime("%m_%d_%H_%M_%S")



preprocesses = [Filters.NLM]
postprocesses = [Filters.NONE]
# range_cors = [Range.NORMALIZE,Range.NORMALIZE,Range.DARK_GAMMA,Range.DARK_GAMMA]
range_cors = [Range.NORMALIZE]
# diff_iterations = [1,2,1,2]
diff_iterations = [1]
# laplacian_scales = [0.75]*4
laplacian_scales = [0.75]
#other_params = [{"alpha":1,"beta":0.5},{"alpha":1,"beta":0.25},{"alpha":1,"beta":0.75},{"alpha":0.5,"beta":0.25}]
# other_params = [{"alpha":1,"beta":0.5}]*4
other_params = [{"alpha":1,"beta":0.5}]


def run_other_methods(run_on_us=False):
    rows = []
    all_keys = None
    for method in Methods:
        print(f'running method {method}')
        if method == Methods.OURS:
            rows_gen=[]
            for i in range(len(preprocesses)):
                average_results = run_by_method(method, run_on_us,i, laplacian_scale=laplacian_scales[i])
                exp_name = f'{laplacian_scales[i]}_{preprocesses[i].name}_{range_cors[i].name}_{diff_iterations[i]}'
                all_keys = average_results.keys()

                rows_gen += [[exp_name, 'us'] + [format(average_results.get(key, ''), '.5f') for key in sorted(all_keys)]]
        else:
            average_results = run_by_method(method, run_on_us)
            exp_name = method.name
            if all_keys == None:
                all_keys = average_results.keys()
                header = ['method', 'dataset'] + list(sorted(all_keys))
                rows = [header]
            rows_gen = [[exp_name, 'us'] + [format(average_results.get(key, ''), '.5f') for key in sorted(all_keys)]]

        rows += rows_gen
    
    with open(os.path.join(results_path, f'metric_results_other_methods_{time}.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)


 
def run_by_method(method, run_on_us_images, ours_index=None, laplacian_scale=1):
    images_path = us_images_path if run_on_us_images else general_images_path
    only_files = [f for f in listdir(images_path) if isfile(join(images_path, f))]
    images_names=[f for f in only_files if ".png" in f]

    average_results = init_avg_results()
    results_list=[]    
    for img_name in tqdm(images_names):
        image = cv2.imread(join(images_path, img_name),0).astype(np.float32) / 255.0
        # if run_on_us_images:
        #     noisy_img = image
        # else:
        #     noisy_img = add_speckle_noise(image).astype(np.float32)
        noisy_img = add_speckle_noise(image).astype(np.float32)
        exp_name = ''
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
                                    laplacian_filter=laplacian * laplacian_scale, 
                                    pyr_levels=4, 
                                    pyr_method=PyrMethod.CV2, 
                                    edge_filter=EdgeFilter.SCHARR,
                                    preprocess_filter=preprocesses[ours_index],
                                    postprocess_filter=postprocesses[ours_index], 
                                    range_correction=range_cors[ours_index],
                                    diffusion_times= diff_iterations[ours_index],
                                    other_params = other_params[ours_index],
                                    log=False)
            exp_name = f'{preprocesses[ours_index].name}_{postprocesses[ours_index].name}_{range_cors[ours_index].name}_{diff_iterations[ours_index]}'
            image = image.squeeze()
            # filtered = np.expand_dims(filtered, 2)

        save_metric_image_results(image, noisy_img, filtered, os.path.join(results_path, f'{img_name}{method.name}_{exp_name}.png'))

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


    for img_name in tqdm(images_names):
        image = cv2.imread(join(images_path, img_name),0).astype(np.float32) / 255.0
        reslts = [('Original', image)]
        for method in Methods:
            if method == Methods.MEDIAN:
                filtered = apply_filter(image, cv2.medianBlur, ksize=5)
            elif method == Methods.GAUSSIAN:
                filtered = apply_filter(image, cv2.GaussianBlur, ksize=(5, 5), sigmaX=0)
            elif method == Methods.NLM:
                filtered = apply_filter(image, restoration.denoise_nl_means, h=0.05, patch_size=7, fast_mode=True)
            elif method == Methods.BILATERAL:
                filtered= apply_filter(image, cv2.bilateralFilter, 5, 2, 2)
            elif method == Methods.SRAD:
                filtered = apply_filter(image, restoration.denoise_tv_bregman, weight=0.1, max_num_iter=10, eps=0.1, isotropic=False)
            elif method == Methods.KUAN:
                filtered= apply_filter(image, restoration.denoise_tv_chambolle, weight=0.03)
            elif method == Methods.OURS:
                continue
            reslts.append((method.name, filtered))
        
        input_image = np.expand_dims(image, 2)
        for i in range(len(preprocesses)):
            filtered = denoise_img(input_image, 
                                    laplacian_filter=laplacian_scales[i]*laplacian, 
                                    pyr_levels=4, 
                                    pyr_method=PyrMethod.CV2, 
                                    edge_filter=EdgeFilter.SCHARR,
                                    preprocess_filter=preprocesses[i],
                                    postprocess_filter=postprocesses[i], 
                                    range_correction=range_cors[i],
                                    diffusion_times= diff_iterations[i],
                                    other_params = other_params[i],
                                    log=False)
            # current_results = reslts + [('Our Method No LF', filtered)]
            current_results = reslts + [('Our Method', filtered)]
            save_multi_method(img_name, current_results, preprocesses[i], postprocesses[i], range_cors[i],diff_iterations[i], laplacian_scales[i])


def save_multi_method(image_name, images, pre, post, range, iter, scale):
    exp_name = f'{scale}_{pre.name}_{post.name}_{range.name}_{iter}'
    fig, axs = plt.subplots(3, 3, figsize=(10,10))

    for i, ax in enumerate(axs.flat):
        if i < len(images):
            ax.imshow(images[i][1], cmap='gray')
            ax.set_title(images[i][0], fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.label_outer()
            plt.imsave(f'{results_path}\\{image_name},{images[i][0]}.png',images[i][1], cmap="gray")

        
    
    plt.tight_layout()
    dir = f'{results_path}\\experiments_{exp_name}_{time}'
    if not os.path.exists(dir):
        os.makedirs(dir)    
    plt.savefig(f'{dir}\\{image_name}')
    plt.close()



def apply_filter(image, filter_func, *args, **kwargs):
    filtered_image = filter_func(image, *args, **kwargs)
    return filtered_image


def run_metrics(laplacian_filter,
                number_layers, 
                edge_filter, 
                preprocess_filter = Filters.NONE, 
                postprocess_filter = Filters.NONE,
                range_correction = Range.HIST_MATCH,
                diffusion_times = 1,
                run_on_us_images=True,
                log_results=True):
    images_path = us_images_path if run_on_us_images else general_images_path
    only_files = [f for f in listdir(images_path) if isfile(join(images_path, f))]
    images_names=[f for f in only_files if ".png" in f]
    average_results=init_avg_results()
    results_list=[]

    for img_name in tqdm(images_names):
        img = cv2.imread(join(images_path, img_name),0).astype(np.float32) / 255.0
        print(img_name)
        results = run_metrics_on_img(img,
                                     laplacian_filter,
                                     number_layers, 
                                     edge_filter, 
                                     preprocess_filter, 
                                     postprocess_filter,
                                     range_correction, 
                                     diffusion_times,
                                     img_name,
                                     run_on_us_images)
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
                       diffusion_times = 1, 
                       img_name=None,
                       run_on_us_images=True):
    if run_on_us_images:
        noisy_img = img
    else:
        noisy_img = add_speckle_noise(img)
    # plt.imsave(f'.\\metrics\\images\\noisy_{img_name}', noisy_img, cmap='gray')
    noisy_img = np.expand_dims(noisy_img, 2)

    clean_image = denoise_img(noisy_img, laplacian_filter, number_layers, PyrMethod.CV2,
                              edge_filter=edge_filter,
                              preprocess_filter=preprocess_filter,
                              postprocess_filter = postprocess_filter,
                              range_correction = range_correction,
                              diffusion_times = diffusion_times,
                              log=False)
    # plt.imsave(f'.\\metrics\\images\\clean_{img_name}', clean_image, cmap='gray')
    # save_image_results(noisy_img, clean_image, f'{results_path}\\final_pair_{img_name}')
                       
    return compute_all_metrics(img,clean_image)


def calculate_iqi(original_image, denoised_image):

    covariance_matrix = np.cov(original_image.flatten(), denoised_image.flatten())
    covariance_value = covariance_matrix[0, 1]  # or covariance_matrix[1, 0]
    variance_img1 = np.var(original_image)
    variance_img2 = np.var(denoised_image)
    mean1 = np.mean(original_image)
    mean2 = np.mean(denoised_image)

    covariance_ratio = covariance_value / (variance_img1 * variance_img2)
    means_ratio = (2*mean1*mean2)/ (mean1**2 + mean2**2)
    variance_ratio = (2*variance_img1*variance_img2)/ (variance_img1**2 + variance_img2**2)

    return covariance_ratio*means_ratio*variance_ratio


def add_speckle_noise(img):
    return random_noise(img, mode='speckle',var=noise_variance)


def compute_all_metrics(origin_img,result_img):
    return {
        # 'mse': meansquareerror(origin_img,result_img),
        # 'signal2noise': signaltonoise(result_img,origin_img),
        # 'psnr': psnr(origin_img,result_img),
        'ssim': ssim(origin_img,result_img,data_range=result_img.max() - result_img.min()),
        # 'sdr': calculate_sdr(result_img, origin_img),
        'cv': calculate_cv(result_img)
        # 'cnr': calculate_contrast_to_noise_ratio(result_img, origin_img),
        # 'iqi': calculate_iqi(origin_img, result_img)
    }


def meansquareerror(src, dst):
    if src.ndim == 3:
        src = src.mean(2)
        #dst = dst.ndim(2)
    mse = np.mean((src - dst) ** 2)
    return mse

def signaltonoise(denoised_image, original_image):
    a = np.asanyarray(denoised_image)
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

def calculate_sdr(denoised_image, original_image):
    residuals = denoised_image - original_image
    sdr = np.std(residuals)
    return sdr


def calculate_cv(image):
    return np.std(image) / np.mean(image)

def calculate_contrast_to_noise_ratio(image, origin):
    noise = origin - image
    mean_contrast = np.mean(image)
    mean_noise = np.mean(noise)
    std_noise = np.std(noise)
    cnr = (mean_contrast - mean_noise) / std_noise
    return cnr

def init_avg_results():
    return {
        # 'mse': 0,
        # 'signal2noise': 0,
        # 'psnr': 0,
        'ssim':0,
        # 'sdr':0,
        'cv': 0,
        # 'cnr': 0,
        # 'iqi':0
        }

def print_results(metrics, avg=False, log_results=True):
    # results_str = f'MSE: {metrics["mse"]},    signal2noise: {metrics["signal2noise"]},    PSNR: {metrics["psnr"]},     SSIM: {metrics["ssim"]},   SDR: {metrics["sdr"]}'
    results_str = f'SSIM: {metrics["ssim"]},   CV: {metrics["cv"]}'
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
    plt.close()


def save_metric_image_results(origin, noisy, denoised, file_name):
    _, axarr = plt.subplots(ncols=3, figsize=(14,14))
    axarr[0].imshow(origin, cmap='gray')
    axarr[0].axis('off')
    axarr[1].imshow(noisy, cmap='gray')
    axarr[1].axis('off')
    axarr[2].imshow(denoised, cmap='gray')
    axarr[2].axis('off')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # laplacian = np.array([0, -1, 0, -1, 4, -1, 0, -1, 0]).reshape((3, 3))
    # number_layers=4
    # run_metrics(laplacian,number_layers) 
    run_other_methods(False)
    # compare_visually()