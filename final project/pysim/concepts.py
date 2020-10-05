import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.ndimage.measurements import center_of_mass
from skimage.measure import label, regionprops
from scipy.signal import medfilt2d


def main():

    sampled, filtered = import_ims()  

    filtered = cv2.fastNlMeansDenoising(sampled, templateWindowSize=3, searchWindowSize=19, h=100.0) 

    filtered[filtered<175] = 0

    peaks = find_peaks2d(filtered[...,0], sampled[...,0])

    print('end')

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
        output[cy, cx] = 1

    # display = filtered_im/255.0

    # display[output>0] = 0

    # plt.imshow(cv2.merge([output+display,display,display]))

    # plt.show()
    
    return output

def import_ims():
    s = cv2.imread('./unfiltgaussians.bmp')
    f = cv2.imread('./gaussians.bmp')

    return s, f

def compare_filts(s, f):

    f1 = cv2.fastNlMeansDenoising(s, templateWindowSize=11, searchWindowSize=15, h=100)

    f2 = cv2.fastNlMeansDenoising(s, templateWindowSize=7, searchWindowSize=19, h=100) 
    f3 = cv2.fastNlMeansDenoising(s, templateWindowSize=3, searchWindowSize=19, h=100) 
    f4 = cv2.fastNlMeansDenoising(s, templateWindowSize=3, searchWindowSize=25, h=100) 

    Y1 = f1[80:150, 300]
    Y2 = f2[80:150, 300]
    Y3 = f3[80:150, 300]
    Y4 = f4[80:150, 300]

    plt.figure(1)
    plt.subplot(2,2,1)
    plt.imshow(f1)
    plt.subplot(2,2,2)
    plt.imshow(f2)
    plt.subplot(2,2,3)
    plt.imshow(f3)
    plt.subplot(2,2,4)
    plt.imshow(f4)

    plt.figure(2)
    plt.subplot(2,2,1)
    plt.plot(Y1)
    plt.subplot(2,2,2)
    plt.plot(Y2)
    plt.subplot(2,2,3)
    plt.plot(Y3)
    plt.subplot(2,2,4)
    plt.plot(Y4)


    plt.show()

main()