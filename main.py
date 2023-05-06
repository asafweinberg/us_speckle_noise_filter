import numpy as np
from metrics import run_metrics
import os
from os import listdir
from os.path import isfile, join
import cv2
from denoise import denoise_img
from enums import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime


images_path=".\\test_images\\images"
results_path=".\\test_images\\output"
metrics_path=".\\metrics\\output"
now = datetime.datetime.now()
time = now.strftime("%m_%d_%H_%M_%S")

def create_metrics_def(laplacian_filter):
    return [
        (laplacian_filter, 4, EdgeFilter.SOBEL_CV2, Filters.NONE, Filters.NONE, Range.hist_match),
        (laplacian_filter, 4, EdgeFilter.SOBEL_CV2, Filters.NONE, Filters.NONE, Range.contrast_stretch),
        (laplacian_filter, 4, EdgeFilter.SOBEL_CV2, Filters.NLM, Filters.NONE, Range.hist_match),
        (laplacian_filter, 6, EdgeFilter.SOBEL_CV2, Filters.NLM, Filters.NONE, Range.hist_match)
    ]

def create_experiments_def(laplacian_filter, images_to_run):
    return [
        (laplacian_filter, 4, EdgeFilter.SOBEL_CV2, Filters.NONE, Filters.NONE, Range.hist_match, images_to_run),
        (laplacian_filter, 4, EdgeFilter.SOBEL_CV2, Filters.NONE, Filters.NONE, Range.contrast_stretch, images_to_run),
        (laplacian_filter, 4, EdgeFilter.SOBEL_CV2, Filters.NLM, Filters.NONE, Range.hist_match, images_to_run),
        (laplacian_filter, 6, EdgeFilter.SOBEL_CV2, Filters.NLM, Filters.NONE, Range.hist_match, images_to_run)
    ]

def calc_metrics(laplacian_filter):
    experiments_def = create_metrics_def(laplacian_filter)
    average_metrics = []
    for i in range(len(experiments_def)):
        exp = experiments_def[i]
        exp_name = get_experiment_name(exp[2],exp[1],exp[3],exp[4],exp[5])
        metrics_results = run_metrics(*experiments_def)
        average_metrics.append((exp_name, metrics_results))

    for (exp_name, metrics_results) in average_metrics:
        with open(os.path.join(results_path, f'metric_results_{time}.txt'), "w") as file:
            file.write(f"{exp_name}: {metrics_results}\n")

def run_many_experiments(laplacian_filter, images_to_run = None):
    experiments_def = create_experiments_def(laplacian_filter, images_to_run)

    experiments_results = []
    clean_images = denoise_multiple_same_method(*experiments_def[0])
    experiments_results = clean_images

    for i in range(1, len(experiments_def)):
        clean_images = denoise_multiple_same_method(*experiments_def[i])
        for key, val in clean_images.items():
            experiments_results[key].append(val[0])


    for image_id, val in experiments_results.items():
        save_grid_images(image_id, val)



def denoise_multiple_same_method(laplacian_filter,
                                 number_layers, 
                                 edge_filter, 
                                 preprocess_filter = Filters.NONE, 
                                 postprocess_filter = Filters.NONE,
                                 range_correction = Range.hist_match,
                                 images_to_run = None):
    only_files = [f for f in listdir(images_path) if isfile(join(images_path, f))]
    images_names = [f for f in only_files if ".png" in f]
    exp_name = get_experiment_name(edge_filter, number_layers, preprocess_filter, postprocess_filter, range_correction)

    if images_to_run:
        images_names = [name for name in images_names if len([id for id in images_to_run if f'({id})' in name])>0]

    clean_images = {name: [] for name in images_names}
    
    print(f'experiment: {exp_name}')

    for img_name in tqdm(images_names):
        img = denoise_single_image(img_name, 
                                   laplacian_filter,
                                   number_layers, 
                                   edge_filter, 
                                   preprocess_filter, 
                                   postprocess_filter,
                                   range_correction)
        clean_images[img_name].append((exp_name, img))
    
    return clean_images
        


def denoise_single_image(img_name,
                         laplacian_filter, 
                         number_layers, 
                         edge_filter, 
                         preprocess_filter = Filters.NONE, 
                         postprocess_filter = Filters.NONE,
                         range_correction = Range.hist_match):
    
    img = cv2.imread(join(images_path, img_name),0).astype(np.float32) / 255.0
    noisy_img = np.expand_dims(img, 2)
    clean_image = denoise_img(noisy_img, 
                              laplacian_filter, 
                              number_layers, 
                              PyrMethod.CV2, 
                              edge_filter=edge_filter,
                              preprocess_filter=preprocess_filter,
                              postprocess_filter = postprocess_filter,
                              range_correction = range_correction,
                              log=True)
    exp_name = get_experiment_name(edge_filter, number_layers, preprocess_filter, postprocess_filter,range_correction)

    exp_path = f'{results_path}\\{exp_name}'
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    current_exp_path = f'{exp_path}\\{time}'
    os.makedirs(current_exp_path)    
    save_image_results(img, clean_image,  f'{current_exp_path}\\{img_name}')
    return clean_image

def get_experiment_name(edge_filter, number_layers, preprocess_filter, postprocess_filter, range_correction):
    exp_name = f'{edge_filter.name}_{number_layers}_pre_{preprocess_filter.name}_post_{postprocess_filter.name}_range_{range_correction.name}'
    return exp_name


def save_grid_images(image_name, images):
    img = cv2.imread(join(images_path, image_name),0).astype(np.float32) / 255.0
    images = [('origin', img)] + images
    fig, axs = plt.subplots(2, 3, figsize=(10,10))

    for i, ax in enumerate(axs.flat):
        if i < len(images):
            ax.imshow(images[i][1], cmap='gray')
            ax.set_title(images[i][0], fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.label_outer()
    
    plt.tight_layout()
    plt.savefig(f'{results_path}\\experiments_{image_name}')


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

    # calc_metrics(laplacian,number_layers) 
    # clean_images(laplacian,number_layers, edge_filter, preprocess_filter=preprocess_filter) 
    run_many_experiments(laplacian, [19])
    calc_metrics(laplacian) 
    #clean_images(laplacian,number_layers, edge_filter, preprocess_filter=preprocess_filter) 
