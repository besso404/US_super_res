from scipy.io import loadmat
from scipy.signal import hilbert
from scipy.interpolate import interp1d
import numpy as np
from matplotlib import pyplot as plt
import cv2

def read_mat_interp():
    mat = loadmat('RFData.mat')
    data = mat['RFData_tot']

    delays = loadmat('Beamforming_workspace.mat')
    del_Tx = np.float32(delays['del_Tx'])
    del_Rx = np.float32(delays['del_Rx'])
    del_Tx = np.array([del_Tx[:,:,0],del_Tx[:,:,4],del_Tx[:,:,7]])

    Rn = np.linspace(0,895, 896)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('./focused_rfdata.mp4', fourcc, 150, (254,203))

    for superframe in range(5):
        cursor = 0
        for m in range(1,200):
            frame = np.zeros((203,254), dtype=np.complex128)
            for na in range(0,3):
                beams = np.zeros((896,128))
                for i in range(0,3):
                    beams[:, :64] += data[cursor:cursor+896, 64:, superframe] 
                    beams[:, 64:] += data[cursor:cursor+896, :64, superframe] 

                    cursor += 896

                for rx in range(128):
                    f = interp1d(Rn, hilbert(beams[:,rx]))
                    terp = del_Rx[:,:,rx]+del_Tx[na,:,:]
                    terp[terp>895] = 895
                    frame += f(terp)
            
            frame = log_scale(np.abs(frame), db=10)
            frame = 255*(frame - frame.min())/(frame.max()-frame.min())
            frame = cv2.cvtColor(np.uint8(frame),cv2.COLOR_GRAY2BGR)
            cv2.imshow('delay and sum',frame)
            cv2.waitKey(2)
            out.write(frame)

    out.release()

    print('1')

def read_mat_ft():
    Zmax  = 50 * 1e-3                      # m
    Trans = 50 * 1e-3                      # m
    FOVz = 20 * 1e-3                       # m
    FOVx = 25 * 1e-3                       # m
    dz = FOVz/896                          # m/px
    dx = FOVx/128                          # m/px
    pitch = Trans/128                      # m/element 
    C = 1540                               # m/sec
    F = 7.5*10e6                           # 1/sec
    wvl = C/F                              # m/wvl
    samples = 4                            # px/wvl

    RxTx = np.zeros((896,128,128), dtype=np.complex128)
    els = pitch*np.linspace(-64,63,128)
    FOVx_mat = np.linspace(-FOVx/2,FOVx/2,128)
    U,V = np.meshgrid(np.linspace(-FOVx/2,FOVx/2,128), np.linspace(-FOVz/2,FOVz/2,896))
    R = np.hypot(U/wvl,V/wvl)
    R[R > 350] = 0
    R[R.nonzero()] = (R.max() - R[R.nonzero()])/R.max()
    
    for el in range(128):
        X, Z = np.meshgrid(np.abs(els[el] - FOVx_mat), np.linspace(Zmax-FOVz, Zmax, 896))
        delays = np.hypot(X, Z)/C + Z/C
        RxTx[:,:,el] = np.exp(-2j*np.pi*delays) * R
    
    # steering = [i * np.pi/180 for i in range(-5, 10, 5)]
    # delays_steer = [dx*np.arange(0,128)*np.sin(theta)/C for theta in steering]
    # delays_steer = [dt_pix*(delay+Rx-delay.min()) for delay in delays_steer]
    # delays_steer = [np.exp(1j*delays) for delays in delays_steer]

    mat = loadmat('RFData.mat')
    data = mat['RFData_tot']

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('./rfdata_focused.mp4', fourcc, 150, (128,896))

    for superframe in range(5):
        cursor = 0
        for m in range(1,200):
            beams = np.zeros((896,128))
            frame = np.zeros((896,128))

            for i in range(0,3):
                    for na in range(0,3):
                        if na==1:
                            beams[:, :64] = data[cursor:cursor+896, 64:, superframe] 
                            beams[:, 64:] = data[cursor:cursor+896, :64, superframe] 

                            beams = np.abs(hilbert2(beams))
                            for el in range(128):
                                S = np.abs(delay_signal(beams, RxTx[:,:,el]))
                                frame += S

                        cursor += 896

            # # envelope = log_scale(frame, db=20)
            frame = 255 * (frame-frame.min())/(frame.max()-frame.min())
            envelope = cv2.cvtColor(np.uint8(frame),cv2.COLOR_GRAY2BGR)
            cv2.imshow('delay and sum',envelope)
            cv2.waitKey(2)
            out.write(envelope)

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
    read_mat_interp()