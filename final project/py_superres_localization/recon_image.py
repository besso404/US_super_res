import numpy as np
from matplotlib import pyplot as plt
import cv2


def post_processing():
    sums = np.load('./peak_sums.npy')
    U = np.load('./U.npy')
    V = np.load('./V.npy')
    db=40
    sums2 = np.uint8(255*(sums+20) / db)

    mask = cv2.adaptiveThreshold(sums2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY,17,0)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    foreground = cv2.dilate(sums2, kernel)
    foreground[mask==0] = 0


    background = cv2.GaussianBlur(sums, (7,7), 0.3)
    background = (background+db)/(background.max()+db)
    foreground = foreground/foreground.max()
    background[mask.nonzero()] += 0.3*foreground[mask.nonzero()]

    plt.figure(1)
    plt.imshow(background, cmap='hot')
    plt.title('Reconstructed and Processed Super-Resolution Image')
    plt.axis('off')


    U[mask==0] = 0
    V[mask==0] = 0

    plt.figure(2)
    show_velocity_map(background, U, V, 1, 1)
    plt.show()

def show_velocity_map(superres, U,V,tx,ty):

    h, w = U.shape

    X, Y = np.meshgrid(np.arange(U.shape[1]), np.arange(U.shape[0]))

    U2 = U * tx
    V2 = V * tx

    U2 = cv2.GaussianBlur(U2, (3,3), 0)
    V2 = cv2.GaussianBlur(V2, (3,3), 0)

    M = np.hypot(U2, V2)

    U2[M>10] = 0
    V2[M>10] = 0
    M[M>10] = 0

    U2[U2.nonzero()] /= M[U2.nonzero()]
    V2[V2.nonzero()] /= M[V2.nonzero()]

    plt.figure(2)
    plt.imshow(superres, cmap='gray')
    plt.quiver(X,Y,U2,V2, M, cmap=plt.cm.nipy_spectral, units='inches', scale=10, angles='xy')
    plt.axis('off')

    clb = plt.colorbar()
    clb.ax.set_title('mm/sec')
    plt.title('Velocity Map')

def log_scale(im, db=1):

    im = (im - im.min()) / (im.max()-im.min())

    b = 1/(10**(db/20))
    a = 1-b

    im = 20 * np.log10(a * im + b)
    return im

post_processing()


print('break')