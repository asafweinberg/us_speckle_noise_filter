from enum import Enum, auto
 
class PyrMethod(Enum):
    MATLAB = auto()
    CV2 = auto()

class EdgeFilter(Enum):
    SOBEL_ND_IMAGE = auto()
    SOBEL_CV2 = auto()
    CANNY = auto()
    SCHARR = auto()
    ICOV = auto()


class Filters(Enum):
    NONE = auto() 
    NLM = auto()
    BILATERAL = auto(),
    KUAN = auto()

class Range(Enum):
    HIST_MATCH = auto() 
    NORMALIZE = auto()
    CONTRAST_STRETCH = auto()

class Methods(Enum):
    GAUSSIAN= auto()
    MEDIAN= auto()
    BILATERAL= auto()
    NLM= auto(),
    KUAN= auto(),
    LEE= auto(),
    SRAD= auto(),
    OURS=auto()