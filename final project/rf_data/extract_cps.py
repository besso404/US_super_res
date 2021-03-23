from scipy.io import loadmat
from scipy.signal import hilbert, hilbert2
from scipy.interpolate import interp1d, RectBivariateSpline, interp2d
import numpy as np
from matplotlib import pyplot as plt
import cv2

def read_mat_interp(record=False):
    mat = loadmat('RFData.mat')
    data = mat['RFData_tot']

    delays = loadmat('Beamforming_workspace.mat')
    del_Tx = np.float32(delays['del_Tx'])
    del_Rx = np.float32(delays['del_Rx'])

    elements = del_Rx.shape[-1]
    imsize = del_Rx.shape[:2]

    depth_pixels = 896
    RF_size = (depth_pixels,elements)

    Rn = np.linspace(0,895, RF_size[0])

    if record:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('./focused_rfdata.mp4', fourcc, 150, (imsize[1], imsize[0]))

    for superframe in range(5):
        cursor = 0
        for m in range(1,200):
            frame = np.zeros(imsize, dtype=np.complex128)
            for na in range(0,3):
                beams = np.zeros(RF_size)
                for i in range(0,3):
                    beams[:, :64] += data[cursor:cursor+depth_pixels, 64:, superframe] 
                    beams[:, 64:] += data[cursor:cursor+depth_pixels, :64, superframe] 

                    cursor += 896

                tx = 3*na+i
                for rx in range(128):
                    f = interp1d(Rn, hilbert(beams[:,rx]))
                    terp = del_Rx[:,:,rx]+del_Tx[:,:,tx]
                    terp[terp>depth_pixels-1] = depth_pixels-1
                    frame += f(terp)
            
            frame = log_scale(np.abs(frame), db=5)
            frame = 255*(frame - frame.min())/(frame.max()-frame.min())
            frame = cv2.cvtColor(np.uint8(frame),cv2.COLOR_GRAY2BGR)
            if record:
                out.write(frame)
            else:
                cv2.imshow('delay and sum',frame)
                cv2.waitKey(2)
    if record:
        out.release()

def read_mat_interp2(record=False):
    mat = loadmat('RFData.mat')
    data = mat['RFData_tot']

    delays = loadmat('Beamforming_workspace.mat')
    del_Tx = np.float32(delays['del_Tx'])
    del_Rx = np.float32(delays['del_Rx'])

    angles = del_Tx.shape[-1]
    elements = del_Rx.shape[-1]
    imsize = del_Rx.shape[:2]

    RxTx = np.zeros((imsize[0], imsize[1], angles, elements))

    for ang in range(angles):
        for el in range(elements):
            RxTx[:,:, ang, el] = del_Tx[:,:,ang]+del_Rx[:,:,el]

    depth_pixels = 896
    RxTx[RxTx>depth_pixels-1] = depth_pixels-1
    RF_size = (depth_pixels,elements)

    Rn = np.linspace(0,895, RF_size[0])
    Tn = np.linspace(0,elements-1, elements)

    Ro = np.linspace(0, depth_pixels-1, imsize[0])
    To = np.linspace(0, elements, imsize[1])
    XX,YY = np.meshgrid(To, Ro)
    XXX = cv2.merge([XX for i in range(elements)])

    if record:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('./focused_rfdata.mp4', fourcc, 150, (imsize[1], imsize[0]))

    for superframe in range(5):
        cursor = 0
        for m in range(1,200):
            frame = np.zeros(imsize, dtype=np.complex128)
            for na in range(0,3):
                beams_in = np.zeros(RF_size)
                beams_out = np.zeros(RF_size, dtype=np.complex128)
                for i in range(0,3):
                    beams_in[:, :64] += data[cursor:cursor+depth_pixels, 64:, superframe] 
                    beams_in[:, 64:] += data[cursor:cursor+depth_pixels, :64, superframe] 

                    cursor += 896

                tx = 3*na+i
                for rx in range(128):
                    beams_out[:,rx] = hilbert(beams_in[:,rx])

                f = RectBivariateSpline(Rn, Tn, np.abs(beams_out))
                frame += np.sum(f(RxTx[:,:,tx,:], XXX, grid=False), axis=-1)
            
            frame = log_scale(np.abs(frame), db=5)
            frame = 255*(frame - frame.min())/(frame.max()-frame.min())
            frame = cv2.cvtColor(np.uint8(frame),cv2.COLOR_GRAY2BGR)
            if record:
                out.write(frame)
            else:
                cv2.imshow('delay and sum',frame)
                cv2.waitKey(2)
    if record:
        out.release()

def read_mat_ft(record=False):
    mat = loadmat('RFData.mat')
    data = mat['RFData_tot']

    delays = loadmat('Beamforming_workspace.mat')
    del_Tx = np.float32(delays['del_Tx'])
    del_Rx = np.float32(delays['del_Rx'])

    elements = del_Rx.shape[-1]
    imsize = del_Rx.shape[:2]
    angles = del_Tx.shape[-1]

    depth_pixels = 896
    RF_size = (depth_pixels,elements)

    RxTx = np.zeros((imsize[0], imsize[1], angles, elements), dtype=np.complex128)

    for ang in range(angles):
        for el in range(elements):
            RxTx[:,:, ang, el] = np.exp(-2j*np.pi*(del_Tx[:,:,ang]+del_Rx[:,:,el]))

    if record:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('./rfdata_focused.mp4', fourcc, 150, (128,896))

    for superframe in range(5):
        cursor = 0
        for m in range(1,200):
            frame = np.zeros(imsize)

            for na in range(0,3):
                beams = np.zeros(RF_size)
                for i in range(0,3):
                    beams[:, :64] += data[cursor:cursor+depth_pixels, 64:, superframe] 
                    beams[:, 64:] += data[cursor:cursor+depth_pixels, :64, superframe] 

                    cursor += 896
                    
                tx = 3*na+i
                beams = cv2.resize(beams, (imsize[1], imsize[0]))
                beams = np.hypot(hilbert2(beams))
                for rx in range(128):
                    S = np.abs(delay_signal(beams, RxTx[:,:,tx,rx]))
                    frame += S


            frame = log_scale(frame, db=5)
            frame = 255*(frame - frame.min())/(frame.max()-frame.min())
            frame = cv2.cvtColor(np.uint8(frame),cv2.COLOR_GRAY2BGR)
            if record:
                out.write(frame)
            else:
                cv2.imshow('delay and sum',frame)
                cv2.waitKey(2)
    if record:
        out.release()

    print('1')

def bpf(rf):
    h, w = rf.shape

    F = np.fft.fftshift(np.fft.fft2(rf))

    peak = np.argmax(np.abs(F[:,64]))
    F_out = np.zeros_like(F)
    crop = F[peak-h//4:peak+h//4,:]
    F_out[h//4:-h//4,:] = crop

    return np.fft.ifft2(np.fft.ifftshift(F_out))

def log_scale(im, db=1):

    im = (im - im.min()) / (im.max()-im.min())

    b = 1/(10**(db/20))
    a = 1-b

    im = 20 * np.log10(a * im + b)
    return (im+db)/db

def delay_signal(signal, delays):
    F = np.fft.fftshift(np.fft.fft2(signal))
    F *= delays
    return np.fft.ifft2(np.fft.ifftshift(F))



if __name__ == "__main__":
    read_mat_interp2()
    #read_mat_ft()