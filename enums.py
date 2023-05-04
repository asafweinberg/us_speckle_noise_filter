from enum import Enum, auto
 
class PyrMethod(Enum):
    MATLAB = auto()
    CV2 = auto()

class EdgeFilter(Enum):
    SOBEL_ND_IMAGE = auto()
    SOBEL_CV2 = auto()
    CANNY = auto()


class Filters(Enum):
    NONE = auto() 
    NLM = auto()
    BILATERAL = auto()

class Range(Enum):
    hist_match = auto() 
    contrast_stretch = auto()