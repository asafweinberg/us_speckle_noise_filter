import matlab.engine
import matlab
from matlab_hadler import *
from pad_image import pad_image_with_bounderies

def generate_blurred_pyramid_func(eng,I,N):
    # Pad with zeros for a rectangular 2-factored image:
    padR, padC = (0, 0)
    I, padR, padC = pad_image_with_bounderies(I)
    BlurredPyramid = [None]*(N)

    BlurredPyramid[0] = np.squeeze(I)
    for i in range(1,N):
        matlab_result = eng.my_impyramid(to_matlab(BlurredPyramid[i-1], expand_dims=True) , 'reduce')
        BlurredPyramid[i] = to_python(matlab_result, expand_dims=False)
    
    return BlurredPyramid, padR, padC

    