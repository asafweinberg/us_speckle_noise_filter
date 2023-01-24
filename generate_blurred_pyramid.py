import matlab.engine
from pad_image import pad_image_with_bounderies

def generate_blurred_pyramid_func(eng,I,N):
    # Pad with zeros for a rectangular 2-factored image:
    padR, padC = (0, 0)
    I = pad_image_with_bounderies(I)
    BlurredPyramid = [None]*(N)

    BlurredPyramid[0] = I
    for i in range(1,N):
        # TODO: exception here- fix conversion to matlab
        BlurredPyramid[i] = eng.my_impyramid(BlurredPyramid[i-1] , 'reduce')
    
    return BlurredPyramid, padR, padC

    