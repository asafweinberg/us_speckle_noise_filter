import matlab
import numpy as np

def to_matlab(A, expand_dims):
    A = np.expand_dims(A,2) if expand_dims else A
    return matlab.double(A.tolist())

def to_python(A, expand_dims, dtype=np.float64):
    np_array =  np.array(A).astype(dtype)
    np_array = np.expand_dims(np_array,2) if expand_dims else np_array
    return np_array