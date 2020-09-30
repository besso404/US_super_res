from scipy.io import loadmat
import cv2
import numpy as np
from scipy.ndimage.measurements import center_of_mass
from skimage.measure import label, regionprops

def get_data():

    first = True

    for i in range (5, 9):

        path = './super_frames/SuperFrameCPS' + str(i) + '.mat'

        mat = loadmat(path)

        if first:
            d = mat['Data']
            first = False
        else:
            d = np.dstack([d, mat['Data']])

    return np.abs(d)

def find_peaks2d(filtered_im, sampled_im):

    # Phase 1 - Isolate all peaks
    peaks = np.zeros_like(filtered_im)

    labels = label(filtered_im>0)

    props = regionprops(labels, intensity_image=sampled_im)

    for obj in props:
        mask = (sampled_im == obj.max_intensity) * (labels == obj.label)
        peaks[mask] = sampled_im[mask]

    # Fill holes
    peaks = cv2.dilate(peaks, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,2)))

    # Phase 2 - Get CoM for each peak
    output = np.zeros_like(filtered_im)
    labels = label(peaks)

    coms = center_of_mass(peaks, labels, range(1,labels.max()))

    for c in coms:

        cy = int(c[0])
        cx = int(c[1])

        output[cy, cx] = 255
         
    return output

def depth_brightener(frame, factor=2):

    h = frame.shape[0]
    w = frame.shape[1]
    scale = np.arange(factor, 0, -factor/h)

    gradient_ = np.array([scale for col in range(w)])
    return gradient_.T

def localization(data):

    im_sum = np.zeros_like(data[:,:,0])
    peak_sums = np.zeros_like(im_sum)
    gradient = depth_brightener(peak_sums)

    no_frames = data.shape[-1]

    lastmax = 1

    for i in range(no_frames):

        if i % 30 == 0 and i > 0:

            thismax = im_sum.max()

            im_sum = (im_sum + im_sum * lastmax/thismax)/(thismax + lastmax) 

            lastmax = thismax

        sample_im = data[:,:,i]

        sample_im = sample_im + sample_im * gradient

        sample_im = np.uint8(255*(sample_im/sample_im.max()))

        filtered = cv2.fastNlMeansDenoising(sample_im, templateWindowSize=15, searchWindowSize=17, h=7.0)

        filtered[filtered<50] = 0
        filtered[filtered>0] = sample_im[filtered>0]

        peaks = find_peaks2d(filtered, sample_im)

        im_sum += peaks
        peak_sums += peaks

        cv2.imshow('sum over time', im_sum)
        cv2.waitKey(50)

    peak_sums = peak_sums**0.3

    w = peak_sums.shape[1]
    h = peak_sums.shape[0]

    peak_sums = cv2.resize(peak_sums, (w*3,h*3))
    cv2.imshow('final sum', peak_sums/peak_sums.max())
    cv2.waitKey(0)

if __name__ == "__main__":
    dataset = get_data()
    localization(dataset)