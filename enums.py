from enum import Enum, auto
 
class PyrMethod(Enum):
    MATLAB = auto()
    CV2 = auto()

class EdgeFilter(Enum):
    SOBEL_ND_IMAGE = auto()
    SOBEL_CV2 = auto()
    CANNY = auto()