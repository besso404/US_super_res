import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
from scipy.ndimage.measurements import center_of_mass
from skimage.util import random_noise
from skimage.measure import label, regionprops
import cv2

def sim_params():

    # Simulation Params
    sim_len = 1000                             #[frames]
    phase1_len = 500                           #[frames]
    FOVx = 5000                                #[um]
    FOVy = 5000                                #[um]
    W = 2500                                   #[um]
    D = 15                                     #[um]                  
    F = 10                                     #[MHz]
    Z1 = 1000                                  #[um]
    Z2 = 1100                                  #[um]
    Z0 = 7500                                  #[um]
    Csound = 1540*1e6                          #[um/sec]
    psf_resolution = 50                        #[pixels?]
    Ncycles = 1
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
        "sim_len":sim_len,         #[frames]
        "phase1_len":phase1_len,   #[frames]
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

def find_peaks2d(filtered_im, sampled_im):

    # Phase 1 - Isolate all peaks
    peaks = np.zeros_like(filtered_im)

    im_erode = cv2.erode(filtered_im, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))

    labels = label(im_erode>0)

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

def simulator(p):

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

    # Init Optical Flow
    u,v = (0,0)
    rec_U = []
    rec_V = []

    # Init Text

    font = cv2.FONT_ITALIC
    t_bottom_left_corner_of_text = (250,200)
    u_bottom_left_corner_of_text = (250,250)
    v_bottom_left_corner_of_text = (250,300)
    font_scale = 0.5
    font_color = (0.6,0.6,0.6)
    line_type = 2
    u_str = "Creating Vessels Mask.."
    v_str = ""

    # Init Morphology Operators

    bubble = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,7))
    cleaner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

    FR = 1/p['dt']
    lastmax = 1

    for t in range(1, p['sim_len']):

        cv2.imshow('Simulation', display)
        c = cv2.waitKey(1)

        if c == 27:
            break
        elif c == 32:
            print('breakpoint')

        mask = np.zeros((p['FOVx'], p['FOVy']), np.float32)

        if t % FR == 0:

            thismax = localizations.max()

            localizations = (localizations + localizations * lastmax/thismax)/(thismax + lastmax) 

            lastmax = thismax

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
        
        sample_im = cv2.filter2D(mask,-1, p['psf'],borderType=cv2.BORDER_CONSTANT)
        mask = cv2.dilate(mask, bubble)

        sample_im = random_noise(sample_im, mode='gaussian')

        sample_im = (255*sample_im/sample_im.max()).astype(np.uint8)

        filtered = cv2.fastNlMeansDenoising(sample_im, templateWindowSize=11, searchWindowSize=15, h=100.0)

        filtered = cv2.erode(filtered, cleaner)
        filtered[filtered<150] = 0
        filtered[filtered>0] = sample_im[filtered>0]

        peaks = find_peaks2d(filtered, sample_im)
        peaks = cv2.merge([peaks, peaks, peaks])
        
        if t == p['phase1_len']:

            vessels = localizations[:,:,0]>0
            last_im = np.copy(filtered) * vessels

        # Optical Flow

        if t > p['phase1_len']:

            frame_gray = np.copy(filtered) * vessels
        
            # Farneback Method
            flow = cv2.calcOpticalFlowFarneback(last_im,
                                                frame_gray, 
                                                None, 
                                                pyr_scale=0.5, 
                                                levels=3, 
                                                winsize=7, 
                                                iterations=3, 
                                                poly_n=5, 
                                                poly_sigma=1.2, 
                                                flags=0)

            dx = flow[...,0]
            dy = flow[...,1]

            # X' = dP / dt
            u = dx * FR
            v = dy * FR

            U = np.abs(u[u != 0])
            if len(U):
                U = np.mean(U)/p['ppm'] 
                rec_U.append(U)
                u_str = "U : %3f, [um/sec]" % (U)
            else:
                u_str = "U : 0 [um/sec]"

            V = v[v != 0]
            if len(V):
                V = np.mean(V)/p['ppm'] 
                rec_V.append(V)
                v_str = "V : %3f, [um/sec]" % (V)
            else:
                v_str = "V : 0 [um/sec]"

            flow = cv2.merge([np.float32(filtered)/filtered.max(), u/u.max(), v/v.max()])
            display[p['FOVx']:, p['FOVy']:, :] = flow

            last_im = np.copy(frame_gray)

        sample_im = (sample_im.astype(np.float32))/sample_im.max()
        
        sample_im = cv2.merge([sample_im, sample_im, sample_im])
        image = background + cv2.merge([mask, mask, mask])
        localizations += peaks

        display[0:p['FOVx'], 0:p['FOVy'], :] = image
        display[0:p['FOVx']:, p['FOVy']:, :] = sample_im/sample_im.max()
        display[p['FOVx']:, :p['FOVy'], :] = localizations

        t_str = "Simulation Step: " + str(t)

        cv2.putText(display,
                    t_str,
                    t_bottom_left_corner_of_text,
                    font,
                    font_scale,
                    font_color,
                    line_type)

        cv2.putText(display,
                    u_str,
                    u_bottom_left_corner_of_text,
                    font,
                    font_scale,
                    font_color,
                    line_type)

        
        cv2.putText(display,
                    v_str,
                    v_bottom_left_corner_of_text,
                    font,
                    font_scale,
                    font_color,
                    line_type)

        if len(exitted_frame1) > 0:
            for b in exitted_frame1:
                bubbles1.pop(b)
                exitted_frame1 = []
        
        if len(exitted_frame2) > 0:
            for b in exitted_frame2:
                bubbles2.pop(b)
                exitted_frame2 = []

    plt.subplot(121)
    plt.hist(rec_U)
    plt.title('Recorded U Velocity')
    plt.xlabel('Velocity [um/sec]')
    plt.ylabel('No. Occurences')
    plt.subplot(122)
    plt.hist(rec_V)
    plt.title('Recorded V Velocity')
    plt.xlabel('Velocity [um/sec]')
    plt.show()
    print('end')
        
        


if __name__ == "__main__":
    p = sim_params()
    simulator(p)
