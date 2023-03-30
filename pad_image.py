import numpy as np
import matlab.engine
 
#function [PaddedImage, padR, padC] = PadImageWithBoundries(I0, N)
#% Pad with zeros for a rectangular 2-factored image:
#    %padFactor = 2^(N+1);
def pad_image_with_bounderies(img):
    padFactor = 10
    # padFactor = int(np.ceil(np.log2(max(*img.shape))))
    padFactor=2**padFactor
    h,w,d = img.shape
    
    padR = padFactor - h
    padC = padFactor - w
    padded_image = np.pad(img,((0,padR),(0,padC),(0,0)),'edge')
    #padded_image = eng.padarray(img, [padR, padC, 0],'replicate','post')

    return padded_image,padR,padC
#%     PaddedImage = padarray(I0,[padR padC 0],'replicate','post');
