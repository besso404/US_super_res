
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d, medfilt2d
from scipy.spatial import distance
from skimage.util import random_noise
from skimage.measure import label, regionprops
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2

def sim_params():

    # Simulation Params
    sim_len = 10000                            #[frames]
    phase1_len = 10                            #[frames]
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
    ppm = 0.1                                  #[pixel/um]
    mu_u = 1040                                #[um/sec]
    std_u = 100                                #[um/sec]
    std_v = 0.5                                #[um/sec]
    FR = 100                                   #[frames/sec]
    noise = 'light'                            #['light' or 'heavy']

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
        "psf_resolution":psf_resolution,
        'noise':noise
    }

def easygauss(X, mu, sigma):

    return np.array([np.exp(-0.5*((X-mu)/sigma)**2)], np.float32)

def elastic_collision(u0, v0):

    u1 = u0
    v1 = -1*v0

    return u1, v1

def group_velocity(track_vect, shape=(500,500)):

    U = np.zeros(shape=shape)
    V = np.zeros(shape=shape)

    paths, num_paths = track_paths(track_vect)
    
    # Calculate velocities of paths
    for p in range(num_paths):

        if len(paths[p])> 6:

            dP = (paths[p][-1] - paths[p][0])/len(paths[p])

            for node in paths[p]:

                py, px = node.astype(np.int32).clip(0, shape[0]-1)

                U[px, py] = dP[0]
                V[px, py] = dP[1]

    return U, V

def track_paths(track_vect):

    endpoints = track_vect[0]
    paths = {}
    path_no = 0

    for t in range(1, len(track_vect)):

        next_points = track_vect[t]

        # Find Euclidean distance
        dists = distance.cdist(next_points, endpoints)

        mins = np.min(dists, axis=1)
        argmins = np.argmin(dists, axis=1)

        for i in range(len(mins)):

            if mins[i] < 6:

                origin = endpoints[argmins[i]]
                node = next_points[i, :]

                new_path = True

                # Find path containing the origin     
                for p in range(path_no):

                    if np.any(np.all(paths[p]==origin, axis=1)):

                        paths[p].append(node)
                        new_path = False

                # Doesn't belong to any path -> New path
                if new_path:

                    paths[path_no] = [origin, node]
                    path_no += 1

        endpoints = [paths[p][-1] for p in range(path_no)]
        if len(endpoints):

            for point in next_points:

                if not np.any(np.all(point==np.array(endpoints), axis=1)):
                    endpoints.append(point)
        else:

            endpoints = track_vect[t]

    return paths, path_no

def calc_speed(p0, p1):

    dP = p1-p0

    # Instantaneous average velocities
    inst_mu = np.mean(dP, axis=0).flatten()

    inst_U = inst_mu[0]
    inst_V = inst_mu[1]

    return (inst_U, inst_V)

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
    
    flow_mask = np.zeros_like(true_image).astype(np.uint8)

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
    U = np.zeros(shape=(p['FOVx'], p['FOVy']))
    V = np.zeros(shape=(p['FOVx'], p['FOVy']))
    U_weights = np.zeros(shape=(p['FOVx'], p['FOVy']))
    V_weights = np.zeros(shape=(p['FOVx'], p['FOVy']))
    
   # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (17,17),
                  maxLevel = 0,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.01))
    
    # Create some random colors
    color = np.random.randint(0,255,(500,3))

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
    arrow = np.zeros((7,7), dtype=np.uint8)
    tracking = []


    for i in range(7):
        for j in range(7):

            if i-j ==3 or i+j == 3:
                arrow[i,j] = 1

    noise_power = p['noise']

    if noise_power == 'heavy':
        noise_var = 0.05
    elif noise_power == 'light':
        noise_var = 0.001
    else:
        # default
        noise_var = 0.01 

    FR = 1/p['dt']

    for t in range(1, p['sim_len']):

        cv2.imshow('Simulation', display)
        c = cv2.waitKey(1)

        if c == 27:
            break
        elif c == 32:
            print('breakpoint')

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
        
        sample_im = cv2.filter2D(mask,-1, p['psf'],borderType=cv2.BORDER_CONSTANT)
        mask = cv2.dilate(mask, bubble)

        sample_im = random_noise(sample_im, mode='gaussian', var=noise_var)

        sample_im = (255*sample_im/sample_im.max()).astype(np.uint8)

        if noise_power == 'heavy':
            filtered = cv2.fastNlMeansDenoising(sample_im, templateWindowSize=3, searchWindowSize=19, h=100.0)
            filtered[filtered<175] = 0
            filtered[filtered>0] = sample_im[sample_im>0]
        else:
            
            filtered = cv2.bilateralFilter(sample_im, d=9, sigmaColor=500, sigmaSpace=25)
            filt_mask = cv2.adaptiveThreshold(filtered,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY, 21,-50)
            filtered[filt_mask==0] = 0
            filtered[filt_mask>0] = sample_im[filt_mask>0]

        peaks = find_peaks2d(filtered, sample_im)
        vessels = localizations[:,:,0]>localizations.max()*0.3

        # Optical Flow
        if t >= p['phase1_len']:

            
            if t % (FR//2) == 0:

                flow_mask = np.zeros_like(true_image).astype(np.uint8)

                if len(tracking):
                    
                    dU, dV = group_velocity(tracking)

                    U_weights[dU.nonzero()] += 1
                    V_weights[dV.nonzero()] += 1

                    U += dU
                    V += dV

                    tracking = []

            frame_gray = cv2.dilate(peaks, arrow)
            
            frame_gray = (255*frame_gray/frame_gray.max()).astype(np.uint8)
            
            flowim = np.copy(localizations)

            frame = cv2.merge([frame_gray, frame_gray, frame_gray])

            p1, st, err = cv2.calcOpticalFlowPyrLK(last_im, frame_gray, p0, None, **lk_params)

            if last_im.max() > 0:

                good_new = p1.reshape(-1,2)
                good_old = p0.reshape(-1,2)
                
                if len(good_new) and len(good_old):

                    u, v = calc_speed(good_old, good_new)

                    tracking.append(good_new)

                # draw the tracks
                for i,(new,old) in enumerate(zip(good_new, good_old)):
                    a,b = new.ravel().astype(int)
                    c,d = old.ravel().astype(int)
                    flow_mask = cv2.line(flow_mask, (a,b),(c,d), color[i].tolist(), 2)
                    frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
                    
                flowim = cv2.add(frame//2,flow_mask).astype(np.float64)

                # Edit display

                flowim = (flowim - flowim.min())/(flowim.max() - flowim.min())
                
                display[p['FOVx']:, p['FOVy']:, :] = flowim 

                u_str = "U : %3f, [um/sec]" % (u * FR/p['ppm'])
                v_str = "V : %3f, [um/sec]" % (v * FR/p['ppm'])

            last_im = np.copy(frame_gray)

        # Use peaks as init points
        pointsy, pointsx = peaks.nonzero()
        p0 = np.array([pointsx, pointsy]).T.reshape(-1, 1, 2).astype(np.float32)

        peaks = cv2.merge([peaks, peaks, peaks])
        localizations += peaks

        sample_im = (sample_im.astype(np.float32))/sample_im.max()
        
        sample_im = cv2.merge([sample_im, sample_im, sample_im])
        image = background + cv2.merge([mask, mask, mask])

        display[0:p['FOVx'], 0:p['FOVy'], :] = image
        display[0:p['FOVx']:, p['FOVy']:, :] = sample_im/sample_im.max()
        display[p['FOVx']:, :p['FOVy'], :] = localizations/localizations.max()

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

    U = U * FR/p['ppm']
    V = V * FR/p['ppm']
    U[U_weights>0] = U[U_weights>0]/U_weights[U_weights>0]
    V[V_weights>0] = V[V_weights>0]/V_weights[V_weights>0]

    localizations = localizations/localizations.max()

    U[localizations[...,0]<0.3] = 0
    V[localizations[...,0]<0.3] = 0

    show_results(U, V, localizations, p['mu_u'])
    analyze_flow(U, p['mu_u'])

def show_results(U, V, super_res, U_real):

    U[U>3000] = 0
    U[U<-3000] = 0

    V[V>3000] = 0
    V[V<-3000] = 0

    magn = (U**2 + V**2)**0.5
    angle = np.arctan2(U, V)
    angle = (angle + np.pi)/2

    labels = label(super_res[...,0]>0.3)
    props_meas = regionprops(labels, intensity_image=U)
    errors = []

    for p in range(len(props_meas)):

        avg_intens = props_meas[p].mean_intensity
        
        err = (np.abs(avg_intens)-U_real)/U_real
        errors.append(np.abs(err))

        error = np.mean(errors)

    print('Errors:')
    print(errors)
    print('Avg Error:')
    print(error)

    fig, ax = plt.subplots(1,3, figsize=(13,7))
    
    im1 = ax[0].imshow(super_res)
    ax[0].set_title('Super-Resolution Image')
    ax[0].get_xaxis().set_visible(False)
    ax[0].get_yaxis().set_visible(False)

    
    im2 = ax[1].imshow(angle, cmap='seismic')
    ax[1].set_title('Velocity Angle')
    ax[1].get_xaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)

    divider1 = make_axes_locatable(ax[1])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)

    ticks1 = np.linspace(0, np.pi,10, endpoint=True)
    tick_labels1 = ["{:5.2f} [rad]".format(i) for i in ticks1]

    cbar1 = fig.colorbar(im2, cax=cax1, orientation='vertical', ticks=ticks1, cmap='seismic')
    cbar1.mappable.set_clim(0, np.pi)
    cbar1.ax.set_yticklabels(tick_labels1)

    im3 = ax[2].imshow(magn, cmap='hot')
    ax[2].set_title('Velocity Magnitude')
    ax[2].get_xaxis().set_visible(False)
    ax[2].get_yaxis().set_visible(False)


    ticks2 = np.linspace(magn.min(),magn.max(),10, endpoint=True)
    tick_labels2 = ["{:5.2f} [um/sec]".format(i) for i in ticks2]

    divider2 = make_axes_locatable(ax[2])
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)

    cbar2 = fig.colorbar(im3, cax=cax2, orientation='vertical', ticks=ticks2, cmap='hot')
    cbar2.mappable.set_clim(magn.min(), magn.max())
    cbar2.ax.set_yticklabels(tick_labels2)  
    
    plt.show()

def analyze_flow(U, U_real):

    magn = np.abs(U)

    U[magn>4000] = 0
    
    plt.hist(U[U.nonzero()], bins=50, alpha=0.55, edgecolor='k', label='recorded velocities')
    plt.axvline(U_real, linewidth=1, color='k', label='expected velocity')
    plt.axvline(-1*U_real, linewidth=1, color='k')

    plt.title('Velocities Measurement Error')
    plt.xlabel('Velocity [um/sec]')
    plt.ylabel('No. Instances')
    plt.legend()

    plt.show()


if __name__ == "__main__":
    p = sim_params()
    simulator(p)
