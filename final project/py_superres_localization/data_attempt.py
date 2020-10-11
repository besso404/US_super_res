
from scipy.io import loadmat
import cv2
import numpy as np
from skimage.measure import label, regionprops
from scipy.signal import medfilt2d

def get_data():

    first = True

    for i in range (5, 6):

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
    peaks = np.zeros_like(sampled_im)

    im_erode = cv2.erode(filtered_im, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))

    labels = label(im_erode>0)

    props = regionprops(labels, intensity_image=sampled_im)

    for obj in props:
        mask = (sampled_im == obj.max_intensity) * (labels == obj.label)
        peaks[mask] = sampled_im[mask]

    # Kill gaps -> Fill holes
    peaks = medfilt2d(peaks, (1,3))
    peaks = cv2.dilate(peaks, cv2.getStructuringElement(cv2.MORPH_RECT, (3,1)))

    # Phase 2 - Use estimated CoM as base for peak-climbing
    peaks2 = np.zeros((filtered_im.shape[0]+2, filtered_im.shape[1]+2), np.uint8)
    labels = label(peaks)
    props2 = regionprops(labels, intensity_image=sampled_im)

    for obj2 in props2:

        c = obj2.weighted_centroid

        cy = int(c[0])
        cx = int(c[1])

        
        l, inp, peaks2, bb = cv2.floodFill(np.float32(filtered_im), peaks2, (cx,cy), newVal=255, loDiff=0, upDiff=30, flags=cv2.FLOODFILL_MASK_ONLY)
        
    peaks2 = peaks2[1:-1, 1:-1]

    # Phase 3 - Get actual CoM    
    output = np.zeros_like(filtered_im, dtype=np.float64)
    labels = label(peaks2)
    props3 = regionprops(labels, intensity_image=sampled_im)

    for obj3 in props3:

        c = obj3.weighted_centroid

        cy = int(c[0])
        cx = int(c[1])
        output[cy, cx] = 255

    # Phase 4 - Get easy peaks from original image
    output[filtered_im==255] = 255

    return output

def depth_brightener(w, h, factor=2):

    scale = np.arange(factor, 0, -factor/h)

    gradient_ = np.array([scale for col in range(w)])
    return gradient_.T

def localization(data):

    w = data.shape[1]
    h = data.shape[0]

    peak_sums = np.zeros((3*h,3*w))
    gradient = depth_brightener(w,h, factor=3)

    no_frames = data.shape[-1]
    

    for i in range(no_frames):

        sample_im = data[:,:,i]

        sample_im = sample_im + sample_im * gradient

        sample_im = cv2.resize(sample_im, (w*3, h*3))

        sample_im = np.uint8(255*(sample_im/sample_im.max()))

        filtered = cv2.fastNlMeansDenoising(sample_im, templateWindowSize=3, searchWindowSize=19, h=7.0)

        filtered[filtered<30] = 0

        peaks = find_peaks2d(filtered, sample_im)

        peak_show = cv2.dilate(peaks, cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5)))

        sample_im[peak_show>0] = 0

        display = cv2.merge([sample_im, sample_im, np.uint8(peak_show)+sample_im])

        peak_sums += peaks

        cv2.imshow('frame analysis', display)
        cv2.waitKey(75)

    peak_sums = peak_sums**0.3

    cv2.imshow('final sum', peak_sums/peak_sums.max())
    cv2.waitKey(0)

    print('done')

if __name__ == "__main__":
    dataset = get_data()
    localization(dataset)