from scipy.io import loadmat
from scipy.signal import hilbert2
import numpy as np
from matplotlib import pyplot as plt
import cv2

def read_mat():
    FOVz = 20 * 1e-3                       # m
    FOVx = 12.5 * 1e-3                     # m
    C = 1540                               # m/sec
    
    # Compute Rx
    Rx = np.zeros((896,128,128))
    dx = FOVx/64

    for el in range(128):
        X, Y = np.meshgrid(np.linspace(el*dx, 2*FOVx - el*dx, 128), np.linspace(0, FOVz, 896))
        delays_dist = np.hypot(X,Y)
        Rx[:,:,el] = 2*delays_dist/C

    weights = [-0.5, 1, -0.5]
    dx = 50 * 1e-3 / 128                   # m/element
    steering = [i * np.pi/180 for i in range(-5, 10, 5)]
    delays_steer = [np.ones((896,128))*np.cumsum(np.ones((128,1)))*np.sin(theta)*dx/C for theta in steering]

    mat = loadmat('RFData.mat')
    data = mat['RFData_tot']

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('./rfdata.mp4', fourcc, 70, (128,896))

    for superframe in range(5):
        cursor = 0
        for m in range(1,200):
            frame = np.zeros((896,128))
            delay_and_sum = np.zeros((896,128))

            for na in range(0,3):
                for i in range(0,3):
                    rf_data = data[cursor:cursor+896, :, superframe]
                    rf_filt = np.abs(bpf(rf_data))

                    frame += rf_filt * weights[i] * delays_steer[na]
                    cursor += 896

            for el in range(128):
                delay_and_sum += frame * Rx[:,:,el]

            envelope = np.abs(hilbert2(delay_and_sum))
            envelope = log_scale(envelope, db=10)
            envelope = cv2.cvtColor(np.uint8(envelope*255),cv2.COLOR_GRAY2BGR)
            out.write(envelope)

    out.release()

    print('1')

def bpf(rf):
    h, w = rf.shape

    F = np.fft.fftshift(np.fft.fft2(rf))
    F_out = np.zeros_like(F)
    crop = F[:h//2,:]
    F_out[h//4:-h//4,:] = crop

    return np.fft.ifft2(np.fft.ifftshift(F_out))
    



def log_scale(im, db=1):

    im = (im - im.min()) / (im.max()-im.min())

    b = 1/(10**(db/20))
    a = 1-b

    im = 20 * np.log10(a * im + b)
    return (im+db)/db

if __name__ == "__main__":
    read_mat()