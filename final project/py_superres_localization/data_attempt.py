from matplotlib import pyplot as plt
from scipy.io import loadmat
import cv2
import numpy as np
from skimage.measure import label, regionprops
from scipy.signal import medfilt2d

def get_data():

    first = True

    for i in range (5, 21):

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

    return output

def tgc_map(h, w, factor=2):

    scale = np.arange(factor, 0, -factor/h)

    gradient_ = np.array([scale for col in range(w)])
    return gradient_.T

def localization(data):

    # Init Rescale and TGC
    w = data.shape[1]
    h = data.shape[0]

    peak_sums = np.zeros((3*h,3*w))
    gradient = tgc_map(h, w, factor=4)

    # Init display text
    no_frames = data.shape[-1]
    
    # Write some Text

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (15, 3*h-15)
    fontScale = 0.5
    fontColor = (200,200,200)
    lineType = 2

    for i in range(no_frames):

        sample_im = data[:,:,i]
        sample_im = sample_im + sample_im * gradient
        sample_im = np.uint8(255*(sample_im/sample_im.max()))
        
        sample_im = cv2.resize(sample_im, (w*3, h*3), interpolation=cv2.INTER_CUBIC)
        
        filtered = cv2.bilateralFilter(sample_im, d=7, sigmaColor=5, sigmaSpace=75)
        
        mask = cv2.adaptiveThreshold(filtered,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,17,-30)

        mask = mask==0

        filtered[mask] = 0

        peaks = find_peaks2d(filtered, sample_im)

        peak_show = cv2.dilate(peaks, cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5)))
        sample_im[peak_show>0] = 0

        display = cv2.merge([sample_im, sample_im, np.uint8(peak_show)+sample_im])

        peak_sums += peaks

        text_str = 'Processed Frame %d/%d'%(i, no_frames)
        cv2.putText(display,text_str, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

        cv2.imshow('frame analysis', display)
        cv2.waitKey(20)

    a = peak_sums**0.2
    b = apply_contrast(peak_sums, relative_thresh=0.1)

    plt.imshow(a, cmap='hot')
    plt.show()

    output_sums = cv2.merge([peak_sums,peak_sums,peak_sums])
    output_a = cv2.merge([a,a,a])
    output_b = cv2.merge([b,b,b])

    cv2.imwrite('./output_sums.png', output_sums)
    cv2.imwrite('./output_a.png', output_a)
    cv2.imwrite('./output_b.png', output_b)

    print('done')

def apply_contrast(im, gamma=0.2, relative_thresh=0.3):

    output = np.copy(im**gamma)
    output[output<output.max()*relative_thresh] = 0

    non_zero = output[output>0]

    output = 255 * (output - non_zero.min())/(non_zero.max()-non_zero.min())
    output[output<0] = 0

    return output




if __name__ == "__main__":
    dataset = get_data()
    localization(dataset)