from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.io import loadmat
import cv2
import numpy as np
from scipy.spatial import distance
from skimage.measure import label, regionprops
from scipy.signal import medfilt2d

def get_data(super_frame_num):

    path = './super_frames/SuperFrameCPS' + str(super_frame_num) + '.mat'

    mat = loadmat(path)

    return np.abs(mat['Data'])

def find_peaks2d(filtered_im, sampled_im, min_dist=5):

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
    found_peaks = []

    for obj3 in props3:

        c = obj3.weighted_centroid

        cy = int(c[0])
        cx = int(c[1])
        output[cy, cx] = 255

        found_peaks.append(np.array([cy,cx]))

    # Phase 4 - Prevent double peaks
    if len(found_peaks):

        dists = distance.pdist(found_peaks)

        dist_mat = np.triu(distance.squareform(dists))

        bad_i, bad_j = ((dist_mat<min_dist) & (dist_mat>0)).nonzero()

        num_bad_pairs = bad_i.shape[0]

        if num_bad_pairs:

            for t in range(num_bad_pairs):

                i = bad_i[t]
                j = bad_j[t]

                cy1, cx1 = found_peaks[i]
                cy2, cx2 = found_peaks[j]

                # Erase bad peaks
                output[cy1, cx1] = 0
                output[cy2, cx2] = 0 

                # Correct peak is their CoM
                cx = (cx1+cx2)//2
                cy = (cy1+cy2)//2

                output[cy,cx] = 255

    return output

def tgc_map(h, w, factor=2):

    scale = np.arange(factor, 0, -factor/h)

    gradient_ = np.array([scale for col in range(w)])
    return gradient_.T

def localization():

    # Determine SupFrame Numbers
    super_frames = range(1,61)

    data0 = get_data(super_frames[0])

    # Init Rescale and TGC
    w0 = data0.shape[1]
    h0 = data0.shape[0]

    scale = 2

    w = int(scale * w0)
    h = int(scale * h0)

    FR = 300
    ppmx = 1/330
    ppmy = 1/110

    peak_sums = np.zeros((h,w))
    gradient = tgc_map(h0, w0, factor=4)
    last_im = np.zeros((h,w), dtype=np.uint8)
    flow_im = np.zeros((h,w,3), dtype=np.uint8)

    # Init display text
    no_frames = data0.shape[-1]
    
    # Write some Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (15, h-15)
    fontScale = 0.5
    fontColor = (200,200,200)
    lineType = 2

    # Init Optical Flow
    u,v = (0,0)
    U = np.zeros(shape=(h, w))
    V = np.zeros(shape=(h, w))
    U_weights = np.zeros(shape=(h, w))
    V_weights = np.zeros(shape=(h, w))
    
   # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (17,17),
                  maxLevel = 0,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.01))

    arrow = np.zeros((7,7), dtype=np.uint8)
    tracking = []
    tracking_interval = 40

    for i in range(7):
        for j in range(7):

            if i-j ==3 or i+j == 3:
                arrow[i,j] = 1

    for s in super_frames:

        data = get_data(s)

        for i in range(0,200):

            sample_im = data[:,:,i]
            sample_im = sample_im + sample_im * gradient
            sample_im = np.uint8(255*(sample_im/sample_im.max()))
            
            sample_im = cv2.resize(sample_im, (w, h), interpolation=cv2.INTER_AREA)
            
            filtered = cv2.bilateralFilter(sample_im, d=7, sigmaColor=150, sigmaSpace=1)
            
            mask = cv2.adaptiveThreshold(filtered,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,17,-35)

            filtered[mask==0] = 0

            filtered = cv2.dilate(filtered, cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5)))

            peaks = find_peaks2d(filtered, sample_im)

            peak_show = cv2.dilate(peaks, cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5)))
            sample_im[peak_show>0] = 0

            display = cv2.merge([sample_im, sample_im, np.uint8(peak_show)+sample_im])

            # Optical Flow
            if s > super_frames[0]:

                if i % tracking_interval == 0:

                    if len(tracking):
                        
                        dU, dV, flow_tracks = group_velocity(tracking, (h, w, 3), tracking_interval)
                        
                        flow_im = np.maximum(flow_im, flow_tracks)  

                        U_weights[dU.nonzero()] += 1
                        V_weights[dV.nonzero()] += 1

                        U += dU
                        V += dV

                        tracking = []

                frame_gray = cv2.dilate(peaks, arrow)
                
                frame_gray = (255*frame_gray/frame_gray.max()).astype(np.uint8)

                p1, st, err = cv2.calcOpticalFlowPyrLK(last_im, frame_gray, p0, None, **lk_params)

                if last_im.max() > 0:

                    good_new = p1.reshape(-1,2)
                    good_old = p0.reshape(-1,2)
                    
                    if len(good_new) and len(good_old):

                        u, v = calc_speed(good_old, good_new)

                        tracking.append(good_new)

                    # Edit display

                    display = np.hstack([display, flow_im])

                    u_str = "U : %3f, [um/sec]" % (u * FR/ppmx)
                    v_str = "V : %3f, [um/sec]" % (v * FR/ppmy)

                last_im = np.copy(frame_gray)

            
            # Use peaks as init points
            pointsy, pointsx = peaks.nonzero()
            p0 = np.array([pointsx, pointsy]).T.reshape(-1, 1, 2).astype(np.float32)

            peak_sums += peaks

            text_str = 'Processed Superframe %d Frame %d/%d'%(s, i, no_frames)
            cv2.putText(display,text_str, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

            cv2.imshow('frame analysis', display)
            cv2.waitKey(20)

    a = apply_contrast(peak_sums, gamma=0.1, relative_thresh=0.4)

    plt.imshow(a, cmap='afmhot')
    plt.show()

    output_sums = cv2.merge([peak_sums,peak_sums,peak_sums])
    output_a = cv2.merge([a,a,a])


    cv2.imwrite('./output_sums.png', output_sums)
    cv2.imwrite('./output_a.png', output_a)
    cv2.imwrite('./output_flow.png', flow_im)

    # Convert pixels/frame to mm/sec
    U = (U * FR/ppmx)/(1000*scale)
    V = (V * FR/ppmy)/(1000*scale)

    U[U_weights>0] = U[U_weights>0]/U_weights[U_weights>0]
    V[V_weights>0] = V[V_weights>0]/V_weights[V_weights>0]

    localizations = peak_sums/peak_sums.max()

    # Use localization mask
    U[a<0.3] = 0
    V[a<0.3] = 0

    show_results(U, V, a)

    print('done')

def apply_contrast(im, gamma=0.2, relative_thresh=0.3):

    output = np.copy(im**gamma)
    output[output<output.max()*relative_thresh] = 0

    non_zero = output[output>0]

    output = 255 * (output - non_zero.min())/(non_zero.max()-non_zero.min())
    output[output<0] = 0

    return output

def group_velocity(track_vect, shape, interval):
    
    U = np.zeros(shape=shape[:-1])
    V = np.zeros(shape=shape[:-1])

    paths, num_paths = track_paths(track_vect)

    # Create some random colors
    colors = np.random.randint(0,255,(num_paths,3))
    tracks = np.zeros(shape=shape, dtype=np.uint8)
    
    # Calculate velocities of paths
    for p in range(num_paths):

        if len(paths[p])> interval//2:

            dP = (paths[p][-1] - paths[p][0])/len(paths[p])

            last_node = None

            for node in paths[p]:

                px, py = node.astype(np.int32)

                px = px.clip(0,shape[1]-1)
                py = py.clip(0,shape[0]-1)

                U[py, px] = dP[0]
                V[py, px] = dP[1]

                if last_node:
                    
                    tracks = cv2.line(tracks, (px,py), (last_node), colors[p].tolist(), 1)

                last_node = (px,py)

    return U, V, tracks

def track_paths(track_vect, min_dist=3):

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

            if mins[i] < min_dist:

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

def show_results(U, V, super_res):

    magn = (U**2 + V**2)**0.5
    angle = np.arctan2(U, V)

    magn[magn>5] = 0
    
    cv2.imwrite('./output_magn.png', cv2.merge([magn, magn, magn]))
    cv2.imwrite('./output_angle.png', cv2.merge([angle, angle, angle]))

    fig, ax = plt.subplots(1,3, figsize=(13,7))
    
    im1 = ax[0].imshow(super_res, cmap='afmhot')
    ax[0].set_title('Super-Resolution Image')
    ax[0].get_xaxis().set_visible(False)
    ax[0].get_yaxis().set_visible(False)

    
    im2 = ax[1].imshow(angle, cmap='hsv')
    ax[1].set_title('Velocity Angle')
    ax[1].get_xaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)

    divider1 = make_axes_locatable(ax[1])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)

    ticks1 = np.linspace(-np.pi, np.pi,10, endpoint=True)
    tick_labels1 = ["{:5.2f} [rad]".format(i) for i in ticks1]

    cbar1 = fig.colorbar(im2, cax=cax1, orientation='vertical', ticks=ticks1, cmap='hsv')
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
    localization()