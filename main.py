import numpy as np
from metrics.metrics import run_metrics


def calc_metrics(laplacian_filter):
    run_metrics(laplacian_filter) 






if __name__ == "__main__":
    
    laplacian = np.array([0, -1, 0, -1, 4, -1, 0, -1, 0]).reshape((3, 3))

    calc_metrics(laplacian) 
