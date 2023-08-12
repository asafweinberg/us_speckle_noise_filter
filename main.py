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
import csv
# from scipy.ndimage import gaussian_filter


# images_path=".\\test_images\\images\\no_black\\no_black_metrics"
images_path=".\\test_images\\images\\very_speckle"
results_path=".\\test_images\\output"
metrics_path=".\\metrics\\output"
now = datetime.datetime.now()
time = now.strftime("%m_%d_%H_%M_%S")

def create_metrics_def(laplacian_filter, run_on_us_images):
    return [
        # [[laplacian_filter, 4, EdgeFilter.SCHARR, Filters.KUAN, Filters.NONE, Range.NORMALIZE, 1, run_on_us_images], [0.5]],
        [[laplacian_filter, 4, EdgeFilter.SCHARR, Filters.NONE, Filters.NONE, Range.DARK_GAMMA, 1, run_on_us_images],[0.75]]
        # (laplacian_filter, 4, EdgeFilter.SCHARR, Filters.NONE, Filters.NONE, Range.DARK_GAMMA, 1, 0.75, True, {"alpha":0.01,"beta":0.5}, run_on_us_images),
        # (laplacian_filter, 4, EdgeFilter.SCHARR, Filters.NONE, Filters.NONE, Range.DARK_GAMMA, 1, 0.75, True, {"alpha":10,"beta":0.5}, run_on_us_images)

    ]

def create_experiments_def(laplacian_filter, images_to_run):
    return [
        # (laplacian_filter, 4, EdgeFilter.SCHARR, Filters.NONE, Filters.NONE, Range.NORMALIZE, 1, 0.75, images_to_run),
        # (laplacian_filter, 4, EdgeFilter.SCHARR, Filters.KUAN, Filters.NONE, Range.NORMALIZE, 1, 0.75, images_to_run),
        # (laplacian_filter, 4, EdgeFilter.SCHARR, Filters.NONE, Filters.NONE, Range.NORMALIZE, 1, 1, images_to_run),
        # (laplacian_filter, 4, EdgeFilter.SCHARR, Filters.KUAN, Filters.NONE, Range.NORMALIZE, 1, 0.5, images_to_run),
        #(laplacian_filter, 4, EdgeFilter.SCHARR, Filters.NONE, Filters.NONE, Range.NORMALIZE, 1, 1, True ,images_to_run),
        #(laplacian_filter, 4, EdgeFilter.SCHARR, Filters.NONE, Filters.NONE, Range.NORMALIZE, 1, 1, False ,images_to_run),
        #(laplacian_filter, 4, EdgeFilter.SCHARR, Filters.NONE, Filters.NONE, Range.NORMALIZE, 1, 0.5, True ,{},images_to_run),
        # (laplacian_filter, 4, EdgeFilter.SCHARR, Filters.NONE, Filters.NONE, Range.DARK_GAMMA, 1, 0.75, False ,{"alpha":1,"beta":0.5}, images_to_run),
        # (laplacian_filter, 4, EdgeFilter.SCHARR, Filters.NONE, Filters.NONE, Range.DARK_GAMMA, 2, 0.75, False ,{"alpha":1,"beta":0.5}, images_to_run),
        # (laplacian_filter, 4, EdgeFilter.SCHARR, Filters.NONE, Filters.NONE, Range.DARK_GAMMA, 1, 0.75, True, {"alpha":1,"beta":0.5}, images_to_run),
        (laplacian_filter, 4, EdgeFilter.SCHARR, Filters.NONE, Filters.NONE, Range.DARK_GAMMA, 1, 0.75, False, {"alpha":1,"beta":0.5}, images_to_run),
        # (laplacian_filter, 4, EdgeFilter.SCHARR, Filters.NONE, Filters.NONE, Range.DARK_GAMMA, 1, 0.75, False, {"alpha":1,"beta":0.5}, images_to_run),
        # (laplacian_filter, 4, EdgeFilter.SCHARR, Filters.NONE, Filters.NONE, Range.NORMALIZE, 1, 0.75, False, {"alpha":1,"beta":0.5}, images_to_run),
        # (laplacian_filter, 4, EdgeFilter.SCHARR, Filters.NONE, Filters.NONE, Range.DARK_GAMMA, 1, 0.75, False, {"alpha":1,"beta":0.5}, images_to_run), 
        # (laplacian_filter, 4, EdgeFilter.SCHARR, Filters.NONE, Filters.NONE, Range.NORMALIZE, 1, 0.75, False, {"alpha":1,"beta":0.5}, images_to_run), 
        # (laplacian_filter, 4, EdgeFilter.SCHARR, Filters.NONE, Filters.NONE, Range.DARK_GAMMA, 1, 0.75, True, {"alpha":0.5,"beta":1}, images_to_run),
        # (laplacian_filter, 4, EdgeFilter.SCHARR, Filters.NONE, Filters.NONE, Range.DARK_GAMMA, 1, 0.75, False, {"alpha":0.5,"beta":1}, images_to_run),!!

        # (laplacian_filter, 4, EdgeFilter.SCHARR, Filters.KUAN, Filters.NONE, Range.NORMALIZE, 1, 0.75, False, {"alpha":1,"beta":0.5}, images_to_run),
        # (laplacian_filter, 4, EdgeFilter.SCHARR, Filters.KUAN, Filters.NONE, Range.NORMALIZE, 1, 0.75, False, {"alpha":0.5,"beta":1}, images_to_run),
        # (laplacian_filter, 4, EdgeFilter.SCHARR, Filters.BILATERAL, Filters.NONE, Range.NORMALIZE, 1, 0.75, False, {"alpha":1,"beta":0.5}, images_to_run),
        # (laplacian_filter, 4, EdgeFilter.SCHARR, Filters.BILATERAL, Filters.NONE, Range.NORMALIZE, 1, 0.75, False, {"alpha":0.5,"beta":1}, images_to_run),
        # (laplacian_filter, 4, EdgeFilter.SCHARR, Filters.NLM, Filters.NONE, Range.NORMALIZE, 1, 0.75, False, {"alpha":1,"beta":0.5}, images_to_run),
        # (laplacian_filter, 4, EdgeFilter.SCHARR, Filters.NLM, Filters.NONE, Range.NORMALIZE, 1, 0.75, False, {"alpha":0.5,"beta":1}, images_to_run),

        # (laplacian_filter, 4, EdgeFilter.SCHARR, Filters.NONE, Filters.NONE, Range.NORMALIZE, 1, 0.75, True, {"alpha":0,"beta":1}, images_to_run),
        # (laplacian_filter, 4, EdgeFilter.SCHARR, Filters.NONE, Filters.NONE, Range.NORMALIZE, 1, 0.75, True, {"alpha":0.1,"beta":1}, images_to_run),
        # (laplacian_filter, 4, EdgeFilter.SCHARR, Filters.NONE, Filters.NONE, Range.NORMALIZE, 1, 0.75, True, {"alpha":0.4,"beta":1}, images_to_run),
        # (laplacian_filter, 4, EdgeFilter.SCHARR, Filters.NONE, Filters.NONE, Range.CONTRAST_STRETCH, 1, 0.75, False, {"alpha":0.1,"beta":1}, images_to_run), !!
        # (laplacian_filter, 4, EdgeFilter.SCHARR, Filters.NONE, Filters.NONE, Range.CONTRAST_STRETCH, 1, 0.75, False, {"alpha":0.2,"beta":1}, images_to_run),
        # (laplacian_filter, 4, EdgeFilter.SCHARR, Filters.NONE, Filters.NONE, Range.NORMALIZE, 2, 0.75, True, {"alpha":1,"beta":0.5}, images_to_run),
        # (laplacian_filter, 4, EdgeFilter.SCHARR, Filters.NONE, Filters.NONE, Range.DARK_GAMMA, 1, 0.75, True, {"alpha":1,"beta":0.5}, images_to_run),
        # (laplacian_filter, 4, EdgeFilter.SCHARR, Filters.NONE, Filters.NONE, Range.DARK_GAMMA, 2, 0.75, True, {"alpha":1,"beta":0.5}, images_to_run)
        #(laplacian_filter, 4, EdgeFilter.SCHARR, Filters.NONE, Filters.NONE, Range.NORMALIZE, 1, 0.75, True ,{"alpha":1,"beta":0.75}, images_to_run),
        #(laplacian_filter, 4, EdgeFilter.SCHARR, Filters.NONE, Filters.NONE, Range.NORMALIZE, 1, 0.75, True ,{"alpha":0.5,"beta":0.25}, images_to_run),
        #(laplacian_filter, 4, EdgeFilter.SCHARR, Filters.NONE, Filters.NONE, Range.NORMALIZE, 1, 1, True ,{}, images_to_run),

    ]

def calc_metrics(laplacian_filter):
    # experiments_def = create_metrics_def(laplacian_filter, False)
    average_metrics = []
    # for i in range(len(experiments_def)):
    #     exp = experiments_def[i][0]
    #     extra = experiments_def[i][1]
    #     exp_name = get_experiment_name(exp[2],exp[1],exp[3],exp[4],exp[5],exp[6], extra[0])
    #     exp[0] *= extra[0]
    #     metrics_results = run_metrics(*exp)
    #     average_metrics.append((exp_name, 'general', metrics_results))

    experiments_def = create_metrics_def(laplacian_filter, False)
    for i in range(len(experiments_def)):
        exp = experiments_def[i][0]
        extra = experiments_def[i][1]
        exp_name = get_experiment_name(exp[2],exp[1],exp[3],exp[4],exp[5],exp[6], extra[0],True,{"alpha":1,"beta":0.5})
        exp[0] = np.multiply(exp[0], extra[0])
        metrics_results = run_metrics(*exp)
        average_metrics.append((exp_name, 'General', metrics_results))

    all_keys = metrics_results.keys()
    header = ['exp_name', 'dataset'] + list(sorted(all_keys))
    rows = [header]
    for exp_name, dataset, d in average_metrics:
        row = [exp_name, dataset] + [format(d.get(key, ''), '.5f') for key in sorted(all_keys)]
        rows.append(row)

    with open(os.path.join(results_path, f'metric_results_{time}.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)

    # with open(os.path.join(results_path, f'metric_results_{time}.txt'), "w") as file:
    #     for (exp_name, metrics_results) in average_metrics:
    #         file.write(f"{exp_name}: {metrics_results}\n")

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
                                 preprocess_filter, 
                                 postprocess_filter,
                                 range_correction,
                                 diffusion_times,
                                 laplacian_scale,
                                 is_lf,
                                 other_params,
                                 images_to_run):
    only_files = [f for f in listdir(images_path) if isfile(join(images_path, f))]
    images_names = [f for f in only_files if ".png" in f]
    exp_name = get_experiment_name(edge_filter, number_layers, preprocess_filter, postprocess_filter, range_correction, diffusion_times, laplacian_scale, is_lf, other_params)

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
                                   range_correction,
                                   diffusion_times,
                                   laplacian_scale,
                                   is_lf,
                                   other_params)
        clean_images[img_name].append((exp_name, img))
    
    return clean_images
        


def denoise_single_image(img_name,
                         laplacian_filter, 
                         number_layers, 
                         edge_filter, 
                         preprocess_filter, 
                         postprocess_filter,
                         range_correction,
                         diffusion_times,
                         laplacian_scale,
                         is_lf,
                         other_params):
    
    img = cv2.imread(join(images_path, img_name),0).astype(np.float32) / 255.0
    noisy_img = np.expand_dims(img, 2)
    clean_image = denoise_img(noisy_img, 
                              laplacian_filter * laplacian_scale, 
                              number_layers, 
                              PyrMethod.CV2, 
                              edge_filter=edge_filter,
                              preprocess_filter=preprocess_filter,
                              postprocess_filter = postprocess_filter,
                              range_correction = range_correction,
                              log=False, 
                              file_name=None,
                              diffusion_times = diffusion_times,
                              is_lf = is_lf,
                              other_params = other_params)
    exp_name = get_experiment_name(edge_filter, number_layers, preprocess_filter, postprocess_filter, range_correction, diffusion_times, laplacian_scale, is_lf,other_params)

    exp_path = f'{results_path}\\{exp_name}'
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    current_exp_path = f'{exp_path}\\{time}'
    if not os.path.exists(current_exp_path):
        os.makedirs(current_exp_path)    
    save_image_results(img, clean_image,  f'{current_exp_path}\\{img_name}') 
    return clean_image

def get_experiment_name(edge_filter, number_layers, preprocess_filter, postprocess_filter, range_correction, diffusion_times, laplacian_scale=1, is_lf=True,other_params={}): 
    try:
        if(len(postprocess_filter)):
            postprocess_str='_'.join([p.name for p in postprocess_filter])
    except:
        postprocess_str=postprocess_filter.name
    exp_name = f'scale_{laplacian_scale}_{preprocess_filter.name}_{postprocess_str}_{range_correction.name}_iter_{diffusion_times}_isLF_{is_lf}_alpha_{other_params["alpha"]}_beta_{other_params["beta"]}'
    return exp_name


def save_grid_images(image_name, images):
    img = cv2.imread(join(images_path, image_name),0).astype(np.float32) / 255.0
    images = [('Original', img)] + images
    ###############################################
    fig, axs = plt.subplots(3, 3, figsize=(10,10))
    for i, ax in enumerate(axs.flat):
        if i < len(images):
            ax.imshow(images[i][1], cmap='gray')
            # if i ==1:
            #     ax.set_title('Our Method - No PP', fontsize=14)
            # if i ==2:
            #     ax.set_title('Our Method', fontsize=14)
            # if i==0:
            ax.set_title(images[i][0], fontsize=6)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.label_outer()
        else:
            ax.axis('off')
    #################################
    # fig, ax = plt.subplots(1, 2)

    # # Display the first grayscale image on the left subplot
    # ax[0].imshow(img, cmap='gray')
    # ax[0].set_title('Original')
    # ax[0].axis('off')

    # # Display the second grayscale image on the right subplot
    # ax[1].imshow(images[1][1], cmap='gray')
    # ax[1].set_title('Denoised')
    # ax[1].axis('off')

    # # Adjust the spacing between the subplots
    # plt.subplots_adjust(wspace=0.1)

    # # Show the figure with the two grayscale images side by side
    # plt.show()
    ###########################################

    plt.tight_layout()
    plt.savefig(f'{results_path}\\experiments_{time}_{image_name}')
    plt.close()


def save_image_results(origin, denoised, file_name):
    _, axarr = plt.subplots(ncols=2, figsize=(14,14))
    axarr[0].imshow(origin, cmap='gray')
    axarr[0].axis('off')
    axarr[1].imshow(denoised, cmap='gray')
    axarr[1].axis('off')
    #plt.savefig(file_name, bbox_inches='tight')
    plt.imsave(file_name,denoised, cmap="gray")


if __name__ == "__main__":
    laplacian = np.array([0, -1, 0, -1, 4, -1, 0, -1, 0]).reshape((3, 3))
    number_layers=4
    edge_filter = EdgeFilter.SOBEL_CV2
    


    ### KENES
    ## Patient treated for a perforated appendicitis with huge amounts of pus and air in the peritoneal cavity
    run_many_experiments(laplacian,[37956]) 

    ## fat liver
    # run_many_experiments(laplacian,[555])

    ## spleen
    # run_many_experiments(laplacian,[16722])







    # run_many_experiments(laplacian,[20788,1616])
    
    #for pic in [17,54,26,93,44,35,46]:
    # for pic in [17,54,93]:
    #    run_many_experiments(laplacian, [pic])
    # calc_metrics(laplacian) 
