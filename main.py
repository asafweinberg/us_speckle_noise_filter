import numpy as np
from metrics.metrics import run_metrics


def calc_metrics(laplacian_filter,number_layers):
    run_metrics(laplacian_filter,number_layers) 






if __name__ == "__main__":
    
    laplacian = np.array([0, -1, 0, -1, 4, -1, 0, -1, 0]).reshape((3, 3))
    number_layers=4
    calc_metrics(laplacian,number_layers) 
