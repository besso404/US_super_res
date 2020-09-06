
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
from scipy.ndimage import median_filter
from skimage.util import random_noise
import cv2

def sim_params():

    # Simulation Params
    FOVx = 5000                                #[um]
    FOVy = 5000                                #[um]
    W = 2500                                   #[um]
    D = 15                                     #[um]                  
    F = 10                                     #[MHz]
    Z1 = 1000                                  #[um]
    Z2 = 1090                                  #[um]
    Z0 = 7500                                  #[um]
    Csound = 1540*1e6                          #[um/sec]
    psf_resolution = 50                        #[pixels?]
    Ncycles = 1
    sim_len = 3000
    ppm = 0.1                                  #[pixel/um]
    mu_u = 1040                                #[um/sec]
    std_u = 100                                #[um/sec]
    std_v = 0.5                                #[um/sec]
    FR = 100                                   #[frames/sec]

    # Calculated Params

    lamda_ = Csound/(F*1e6)                    #[um]
    FWHM_lat = 0.886*ppm*lamda_*Z0/W           #[pixel]
    sigma_lat = FWHM_lat/2.355                 #[pixel]
    FOVx_ = int(np.floor(FOVx*ppm))            #[pixel]
    FOVy_ = int(np.floor(FOVy*ppm))            #[pixel]
    FWHM_ax = Ncycles*lamda_*ppm/2             #[pixel]
    sigma_ax = FWHM_ax/2.355                   #[pixel]
    up_lim1 = int(np.floor((Z1-(D/2))*ppm))    #[pixel]
    down_lim1 = int(np.ceil((Z1+(D/2))*ppm))   #[pixel]
    up_lim2 = int(np.floor((Z2-(D/2))*ppm))    #[pixel]
    down_lim2 = int(np.ceil((Z2+(D/2))*ppm))   #[pixel]
    dt = 1/FR                                  #[sec/frame]
    sigma_y = (D/2)**0.5                       #[um^0.5]
 
    psf_lat = easygauss(np.linspace(-psf_resolution//2+1,psf_resolution//2), 0, sigma_lat)
    psf_ax = easygauss(np.linspace(-psf_resolution//2+1,psf_resolution//2), 0, sigma_ax)
    psf = convolve2d(psf_ax.T,psf_lat)

    return {
        "FOVx":FOVx_,              #[pixel]
        "FOVy":FOVy_,              #[pixel]
        "ppm":ppm,                 #[pixel/um]
        "dt":dt,                   #[sec/frame]
        "Z1":Z1,                   #[um]    
        "Z2":Z2,                   #[um]
        "D":D,                     #[um]
        "sigma_y":sigma_y,         #[um^0.5]
        "up_lim1":up_lim1,         #[pixel]
        "down_lim1":down_lim1,     #[pixel]
        "up_lim2":up_lim2,         #[pixel]
        "down_lim2":down_lim2,     #[pixel]
        "mu_u":mu_u,               #[um/sec]
        "std_u":std_u,             #[um/sec]
        "std_v":std_v,             #[um/sec]
        "psf":psf,             
        "psf_resolution":psf_resolution
    }

def easygauss(X, mu, sigma):

    return np.array([np.exp(-0.5*((X-mu)/sigma)**2)], np.float32)

def elastic_collision(u0, v0):

    u1 = u0
    v1 = -1*v0

    return u1, v1

def calc_speed(p1_, p0_, fr_, ppm_):

    dp = p1_ - p0_
    
    u = np.mean(dp[:,0])*fr_/ppm_
    v = np.mean(dp[:,1])*fr_/ppm_

    return (u,v)


def simulator(p, sim_len=100000):

    # Init Background

    sample = np.zeros((p['FOVx'], p['FOVy'], 3), np.float32)
    true_image = np.zeros((p['FOVx'], p['FOVy']), np.float32)
    last_im = np.copy(true_image).astype(np.uint8)

    localizations = np.copy(sample)
    
    true_image[p['up_lim1']:p['down_lim1'],:] = 1
    true_image[p['up_lim2']:p['down_lim2'],:] = 1

    true_image = cv2.merge([true_image*0.1, true_image*0.1, true_image])
    image = np.zeros((p['FOVx'], p['FOVy'], 3), np.float32)
    sample_im = np.copy(image)
    background = np.copy(true_image)

    displaytop = np.hstack([true_image, sample])
    displaybottom = np.hstack([image, sample])

    display = np.vstack([displaytop, displaybottom])

    cv2.imshow('Simulation', display)
    cv2.waitKey(1)

    # Bubbles

    bubbles1 = [{
        'y':np.random.normal(p['Z1'], p['sigma_y'])*p['ppm'],
        'x':0,
        'u':p['mu_u']*p['ppm']*p['dt'],
        'v':np.random.normal(0, p['std_v'])*p['ppm']*p['dt'],
        't0':1
        }]

    bubbles2 = [{
        'y':np.random.normal(p['Z2'], p['sigma_y'])*p['ppm'],
        'x':p['FOVx']-1,
        'u':-1*p['mu_u']*p['ppm']*p['dt'],
        'v':np.random.normal(0, p['std_v'])*p['ppm']*p['dt'],
        't0':1
        }]
    exitted_frame1 = []
    exitted_frame2 = []
    # Init Morphology Operators

    strel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    strel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    bubble = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,7))

    # Init Optical Flow

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (21,21),
                    maxLevel = 1,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.05))

    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                            qualityLevel = 0.3,
                            minDistance = 7,
                            blockSize = 7 )

    # Create some random colors
    color = np.random.randint(0,255,(500,3))

    # Start simulation

    FR = 1/p['dt']
    for t in range(1, sim_len):

        mask = np.zeros((p['FOVx'], p['FOVy']), np.float32)

        if t % FR == 0:

            bubbles1.append({
            'y':np.random.normal(p['Z1'], p['sigma_y'])*p['ppm'],
            'x':0,
            'u':p['mu_u']*p['ppm']*p['dt'],
            'v':np.random.normal(0, p['std_v'])*p['ppm']*p['dt'],
            't0':t
            })
            bubbles2.append({
            'y':np.random.normal(p['Z2'], p['sigma_y'])*p['ppm'],
            'x':p['FOVx']-1,
            'u':-1*p['mu_u']*p['ppm']*p['dt'],
            'v':np.random.normal(0, p['std_v'])*p['ppm']*p['dt'],
            't0':t
            })

        for b in range(len(bubbles1)):

            bubbles1[b]['y'] = bubbles1[b]['y'] + (t-bubbles1[b]['t0'])*bubbles1[b]['v']

            bubbles1[b]['x'] = (t - bubbles1[b]['t0'])*bubbles1[b]['u']

            if bubbles1[b]['y'] < p['up_lim1'] + abs(bubbles1[b]['v']) or bubbles1[b]['y'] > p['down_lim1'] - abs(bubbles1[b]['v']):
                bubbles1[b]['u'], bubbles1[b]['v'] = elastic_collision(bubbles1[b]['u'], bubbles1[b]['v'])

            if bubbles1[b]['x'] > p['FOVx'] - bubbles1[b]['u']:
                exitted_frame1.append(b)

            mask[int(np.ceil(bubbles1[b]['y'])), int(np.min([p['FOVx']-1, round(bubbles1[b]['x'])]))] = 1


        for b in range(len(bubbles2)):

            bubbles2[b]['y'] = bubbles2[b]['y'] + (t-bubbles2[b]['t0'])*bubbles2[b]['v']

            bubbles2[b]['x'] = bubbles2[b]['x'] + bubbles2[b]['u']

            if bubbles2[b]['y'] < p['up_lim2'] + abs(bubbles2[b]['v']) or bubbles2[b]['y'] > p['down_lim2'] - abs(bubbles2[b]['v']):
                bubbles2[b]['u'], bubbles2[b]['v'] = elastic_collision(bubbles2[b]['u'], bubbles2[b]['v'])

            if bubbles2[b]['x'] < 0 + bubbles2[b]['u']:
                exitted_frame2.append(b)

            mask[int(np.ceil(bubbles2[b]['y'])), int(np.max([0, round(bubbles2[b]['x'])]))] = 1
        
        sample_im = cv2.filter2D(mask,-1, p['psf'])
        mask = cv2.dilate(mask, bubble)

        # sample_im = random_noise(sample_im, mode='gaussian')

        sample_im = (255*sample_im/sample_im.max()).astype(np.uint8)

        input_im = cv2.copyMakeBorder(sample_im, 25, 25, 25, 25, cv2.BORDER_CONSTANT, None, 0)

        input_im = (255*input_im).astype(np.uint8)

        corr = cv2.matchTemplate(input_im, (255*p['psf']).astype(np.uint8), cv2.TM_CCOEFF)

        corr = corr[1:, 1:]

        peaks = (255*corr/corr.max()).astype(np.uint8)

        filt = (corr > 180) & (sample_im > 250)
        peaks[filt] = 255 
        peaks[~filt] = 0 
        
        # peaks = median_filter(peaks, size=3)

        flowim = cv2.dilate(peaks, bubble)

        found_bubbles = flowim == 255

        flowim[found_bubbles] = sample_im[found_bubbles]

        sample_im = cv2.merge([sample_im, sample_im, sample_im])
        image = background + cv2.merge([mask, mask, mask])

        # Optical Flow

        frame_gray = np.copy(flowim)
        peaks = cv2.merge([peaks, peaks, peaks])
        frame = cv2.merge([frame_gray, frame_gray, frame_gray])
        flow_mask = np.zeros_like(peaks).astype(np.uint8)

        
        if t % 10 == 0 or t == 1:
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask = None, **feature_params)
        
        p1, st, err = cv2.calcOpticalFlowPyrLK(last_im, frame_gray, p0, None, **lk_params)

        if p1 is not None:

            # Select good points
            good_new = p1[st==1]
            good_old = p0[st==1]

            u, v = calc_speed(good_new, good_old, FR, p['ppm'])
            print(u,v)

            # draw the tracks
            for i,(new,old) in enumerate(zip(good_new, good_old)):
                a,b = new.ravel()
                c,d = old.ravel()
                flow_mask = cv2.line(flow_mask, (a,b),(c,d), color[i].tolist(), 2)
                frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
            
        flowim = cv2.add(frame//2,flow_mask)
        
        p0 = good_new.reshape(-1,1,2)
        
        last_im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Edit display

        flowim = flowim.astype(np.float64)

        flowim = (flowim - flowim.min())/(flowim.max() - flowim.min())
        localizations += peaks
        superes = (localizations - localizations.min())/(localizations.max() - localizations.min())

        sample_im = (sample_im.astype(np.float32))/sample_im.max()
        input_im = (input_im.astype(np.float32))/input_im.max()

        display[0:p['FOVx'], 0:p['FOVy'], :] = image
        display[0:p['FOVx']:, p['FOVy']:, :] = sample_im/sample_im.max()
        display[p['FOVx']:, :p['FOVy'], :] = superes
        display[p['FOVx']:, p['FOVy']:, :] = flowim

        if len(exitted_frame1) > 0:
            for b in exitted_frame1:
                bubbles1.pop(b)
                exitted_frame1 = []
        
        if len(exitted_frame2) > 0:
            for b in exitted_frame2:
                bubbles2.pop(b)
                exitted_frame2 = []

        cv2.imshow('Simulation', display)
        c = cv2.waitKey(1)

        if c == 27:
            break
        
        


if __name__ == "__main__":
    p = sim_params()
    simulator(p)