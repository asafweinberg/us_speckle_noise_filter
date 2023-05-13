import numpy as np
# from scipy.fftpack import dst, idst

def poisson_solver_py(gx, gy, boundary_image):
    H, W = boundary_image.shape
    gxx = np.zeros((H, W))
    gyy = np.zeros((H, W))
    j = np.arange(0, H-1)
    k = np.arange(0, W-1)
    # Laplacian
    gyy[1:,:] = gy[1:,:] - gy[:-1,:]
    gxx[:,1:] = gx[:,1:] - gx[:,:-1]
    f = gxx + gyy
    del j, k, gxx, gyy
    # boundary image contains image intensities at boundaries
    boundary_image[1:-1, 1:-1] = 0
    # compute boundary points
    # j = np.arange(1, H-1)
    # k = np.arange(1, W-1)
    f_bp = np.zeros((H, W))
    f_bp[1:-1,1:-1] = -4*boundary_image[1:-1,1:-1] + boundary_image[1:-1,2:] + boundary_image[1:-1,:-2] + boundary_image[0:-2,1:-1] + boundary_image[2:,1:-1]
    f1 = f - f_bp  # subtract boundary points contribution
    del f_bp, f
    # DST Sine Transform algo starts here
    f2 = f1[1:-1, 1:-1]
    del f1
    # compute sine transform
    tt = dst(f2)
    f2sin = dst(tt.T).T
    del f2, tt
    # compute Eigen Values
    x, y = np.meshgrid(np.arange(1, W-1), np.arange(1, H-1))
    denom = (2*np.cos(np.pi*x/(W-1))-2) + (2*np.cos(np.pi*y/(H-1)) - 2)
    del x, y
    # divide
    f3 = f2sin/denom
    del f2sin, denom
    # compute Inverse Sine Transform
    tt = idst(f3)
    del f3
    img_tt = idst(tt.T).T
    del tt
    # put solution in inner points; outer points obtained from boundary image
    img_direct = boundary_image.copy()
    img_direct[1:-1, 1:-1] = img_tt
    return img_direct


def dst(a, n=None):
    if n is None:
        n = len(a)
    if len(a.shape) == 1:
        if a.shape[0] > 1:
            do_trans = True
        else:
            do_trans = False
        a = a.reshape((-1, 1))
    else:
        do_trans = False
    m = a.shape[1]
    if a.shape[0] < n:
        aa = np.zeros((n, m))
        aa[:a.shape[0], :] = a
    else:
        aa = a[:n, :]
    y = np.zeros((2*(n+1), m))
    y[1:n+1, :] = aa
    y[n+2:2*(n+1), :] = -np.flipud(aa)
    yy = np.fft.fft(y, axis=0)
    b = yy[1:n+1, :]/(-2j)
    if np.isreal(a).all():
        b = b.real
    if do_trans:
        b = b.T
    return b

def idst(a, n=None):
    if n is None:
        if len(a.shape) == 1:
            n = len(a)
        else:
            n = a.shape[0]
    nn = n+1
    b = 2/nn*dst(a, n)
    return b