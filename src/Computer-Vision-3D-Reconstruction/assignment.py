import glm
import numpy as np
import cv2
import cv2 as cv
from sklearn.mixture import GaussianMixture as gm
from joblib import Parallel, delayed
import time
import os


# region Configuration
block_size   = 1.2
WORLD_SCALE  = 20
VOXEL_STEP   = 16
CAM_IDS      = [1, 2, 3, 4]

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', 'data'))

_cam_configs  = None
_bg_models    = None
_lookup_table = None
_voxel_world  = None
_in_front     = None
_cam_depths   = None
_frame_idx    = 0

# endregion

# region Helpers
# Loads camera calibration parameters (K, dist, rvec, tvec) from each camera's config.xml and caches the result.
def _ensure_cam_configs():
    global _cam_configs
    if _cam_configs is not None:
        return _cam_configs
    _cam_configs = []
    for cid in CAM_IDS:
        p = os.path.join(DATA_DIR, f'cam{cid}', 'config.xml')
        fs = cv2.FileStorage(p, cv2.FILE_STORAGE_READ)
        _cam_configs.append((
            fs.getNode("camera_matrix").mat(),
            fs.getNode("distortion_coefficients").mat(),
            fs.getNode("rotation_vector").mat(),
            fs.getNode("translation_vector").mat(),
        ))
        fs.release()
    return _cam_configs


# Builds a per-camera background model using create_background_model_simple (Cell 6) and caches the result.
def _ensure_bg_models():
    global _bg_models
    if _bg_models is not None:
        return _bg_models
    print("Building background models …")
    for cid in CAM_IDS:
        create_background_model_mog(cid)
    _bg_models = bg_models
    return _bg_models


# ## Task 2: Background Subtraction (Vinn)
# Focus: HSV thresholding and morphology on `background.avi` and `video.avi`.

# ## Sources
#
# - https://en.wikipedia.org/wiki/Foreground_detection
# - https://en.wikipedia.org/wiki/Moving_average
# - https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
# - https://github.com/pransen/ComputerVisionAlgorithms/blob/master/GaussianMixtureModelling/README.md

#region Helper Functions for retrieving video material
def get_background_video(cam_id):
    return cv.VideoCapture(f'../data/cam{cam_id}/background.avi')

def get_video(cam_id):
    return cv.VideoCapture(f'../data/cam{cam_id}/video.avi')
#endregion

bg_models = []
basic_bg = []

# region Background Modelling using a MOG method
# source used: https://github.com/pransen/ComputerVisionAlgorithms/blob/master/GaussianMixtureModelling/README.md
def create_background_model_mog(cam_id):
    """Function that models each Pixel as a mixture of Gaussians based on HSV values"""
    video = get_background_video(cam_id)

    frames = []
    while True:
        ret, frame = video.read()

        # 1. Extract frames from the video.
        if ret:
            #grab the first 100 frames
            frames.append(cv.cvtColor(frame, cv.COLOR_BGR2HSV))
            
            if len(frames) == 100:
                break
        else:
            video.release()
            return
        
    # 2. Stack the frames in an array where the final array dimensions will be (num_frames, image_width, image_height, num_channels)
    frame_stack = np.stack(frames, axis=0)
    # Flatten stack to avoid nested loops
    flat_stack = frame_stack.reshape(frame_stack.shape[0], -1, 3)
    
    model = []
        
    print("Modeling background for camera", cam_id)
    # time_start = time.perf_counter()
    
    # 4. For each point characterized by the x-coordinate, the y-coordinate and the channel, 
    # model the HSV value as a mixture of Gaussians using SciKit's GaussianMixture library.  
    model = Parallel(n_jobs=-1)(delayed(gaussian_fit)(flat_stack[:,x,:]) 
                                for x in range(frame_stack.shape[1] * frame_stack.shape[2]))
    
    # time_end = time.perf_counter()
    # duration = time_end - time_start
    # print("Finished modeling background for camera", cam_id, "in", duration, "seconds")
    
    bg_models.append(model)

def gaussian_fit(pixel):
    """Run SciKit's GaussianMixture.fit function with a pixel (ideally an array of its recorded HSV values)"""
    return gm(n_components=1, covariance_type="full", reg_covar = 1).fit(pixel)
# endregion

MIN_WEIGHT = 2.5
MAX_WEIGHT = 10.0

# Create a foreground mask per camera using the background model made earlier
def get_foreground_mask(frame, mean, covar, thresholds = [15, 15, 15], thresholding = False):
    """Creates a foreground mask per frame, using the calculated means and covariance matrix, as well as the HSV thresholds"""

    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
    weight_h, weight_v, weight_s = thresholds
    
    difference = frame_hsv.astype(np.float32) - mean
    diff_h, diff_s, diff_v = cv.split(difference)
    
    # Calculate the Mahalanobis Distance between pixel value X and model means 
    # source: https://anandksub.dev/blog/gmm_mahalanobis
    
    covar_h = covar[:,:,0] # Top Part of Covariance Metrix    [VarH, CovHS, CovHV]
    covar_s = covar[:,:,1] # Middle Part of Covariance Matrix [CovSH, VarS, CovSV]
    covar_v = covar[:,:,2] # Bottom Part of Covariance MAtrix [CovVH, CovVS, VarV]

    # Calculate the Mahalanobis distance for each channel separately. 
    # Square Root of (value X - mu) * inverse_covariance * value X - mu 
    dist_h = np.sqrt(diff_h * (covar_h[:,:,0]*diff_h + covar_h[:,:,1]*diff_s + covar_h[:,:,2]*diff_v))
    dist_s = np.sqrt(diff_s * (covar_s[:,:,0]*diff_h + covar_s[:,:,1]*diff_s + covar_s[:,:,2]*diff_v))
    dist_v = np.sqrt(diff_v * (covar_v[:,:,0]*diff_h + covar_v[:,:,1]*diff_s + covar_v[:,:,2]*diff_v))

    # Threshold each channel separately with weights adapted beforehand
    foreground_h = dist_h > weight_h
    foreground_s = dist_s > weight_s
    foreground_v = dist_v > weight_v

    # Detects foreground from both a hue + saturation change (shade has changed and became darker)
    # or with a hue + value change (shade has changed and became brighter)
    foreground = (foreground_h & foreground_s) | (foreground_h & foreground_v)
    mask = (foreground * 255).astype(np.uint8)
    
    if not thresholding:
        mask = post_process(mask)
    return mask

def post_process(mask):
    # Post Processing
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
    bigger_kernel = cv.getStructuringElement(cv.MORPH_RECT, (9,9))
    
    # There's still quite a bit of noise left after morphological operations
    # Use OpenCV's function to find and label all components in image depending on the connectivity
    
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel) # Erode => Dilate (remove surrounding noise)

    # An extra step to remove surrounding noise that was left after Morph_Open, this is also to preserve the biggest figure (the horse)
    blobs = cv2.connectedComponentsWithStats(mask, 8, cv.CV_32S)
    num, labels, stats, centroids = blobs
    biggest_blob = stats[1:, cv.CC_STAT_AREA].max() # Determine the biggest area (often the horse)
    

    # Go through labels, if label area is smaller than blob, set it to 0 since it is noise
    for i in range(1, num):
        if stats[i, cv.CC_STAT_AREA] < biggest_blob:
            mask[labels == i] = 0
    
    # There are holes left in the figure itself, so with a bigger kernel, Morph_Close is performed to fill in these holes
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, bigger_kernel) # Dilate => Erode (fill in inner gaps)

    return mask

# CHOICE 2: Automated thresholding function
def optimize_thresholds(frame, mean, covar, iterations = 30):
    """Tries out random threshold values using the first frame as a reference. """
    
    most_optimal_thresholds = [0, 0, 0]
    lowest_score = float('inf')

    # Iterate based on given iterations param
    for i in range(iterations):
        min_weights = [MIN_WEIGHT, MIN_WEIGHT, MIN_WEIGHT] * 2
        max_weights = [MAX_WEIGHT, MAX_WEIGHT, MAX_WEIGHT * 2]

        weights = [np.random.uniform(min_weights[0], max_weights[0]), np.random.uniform(min_weights[1], max_weights[1]), np.random.uniform(min_weights[2], max_weights[2])]
        
        mask = get_foreground_mask(frame, mean, covar, weights, True)
        
        # compute noise in the foreground mask using the current thresholds
        noise = np.sum(mask == 0) / (mask.shape[0] * mask.shape[1])
        
        # if the noise is less than the lowest score, update the most optimal thresholds and lowest score
        if noise < lowest_score:
            lowest_score = noise
            
            # update weights for next iteration to be around the current optimal weights

            for i, weight in enumerate(weights):
                min_weights[i] = weight - 2
                max_weights[i] = weight + 2

            most_optimal_thresholds = weights
        
    return most_optimal_thresholds

# Unpacks the pixel's model values into the mean and covariance
def unpack_gaussian_model(frame, model):
    """Get the mean and covariance values from a pixel model"""
    mean = []
    covariance = []
    
    for pixel in model:
        mean.append(pixel.means_)
        covariance.append(pixel.covariances_)
        
    # Reshape to match image dimensions W x H
    x, y, c = frame.shape
    mean = np.reshape(mean, (x, y , c))
    covariance = np.reshape(covariance, (x, y, 3, 3))
    cov_inv = np.linalg.inv(covariance[:,:])
    return mean, cov_inv

# Builds and caches a lookup table that projects every 3D voxel to 2D pixel coordinates per camera.
# World coords: X,Y on floor, Z<0 above floor. Visualiser coords: X,Z on floor, Y up.
def _ensure_lut(width, height, depth):
    global _lookup_table, _voxel_world, _in_front, _cam_depths
    if _lookup_table is not None:
        return _lookup_table, _voxel_world

    cfgs = _ensure_cam_configs()

    half_w = (width / 2) * WORLD_SCALE
    half_d = (depth / 2) * WORLD_SCALE
    h_max  = height * WORLD_SCALE * 2

    xs = np.arange(-half_w, half_w, VOXEL_STEP, dtype=np.float64)
    ys = np.arange(-half_d, half_d, VOXEL_STEP, dtype=np.float64)
    zs = np.arange(-h_max, 0, VOXEL_STEP, dtype=np.float64)

    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing='ij')
    _voxel_world = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

    N = len(_voxel_world)
    print(f"Building LUT: {len(xs)}×{len(ys)}×{len(zs)} = {N:,} voxels "
        f"(step {VOXEL_STEP} mm)")

    _lookup_table = {}
    _in_front = {}
    _cam_depths = {}
    for i, (K, d, rv, tv) in enumerate(cfgs):
        proj, _ = cv2.projectPoints(_voxel_world, rv, tv, K, d)
        _lookup_table[i] = proj.reshape(-1, 2)

        R, _ = cv2.Rodrigues(rv)
        P_cam = (_voxel_world @ R.T) + tv.T
        _in_front[i] = P_cam[:, 2] > 0
        _cam_depths[i] = P_cam[:, 2].astype(np.float64)

        print(f"  cam{CAM_IDS[i]} – {np.sum(_in_front[i]):,}/{N:,} in front")

    return _lookup_table, _voxel_world

# endregion

# region Public API

# Generates a checkerboard floor grid of alternating black and white tiles.
def generate_grid(width, depth):
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x * block_size - width / 2,
                        -block_size,
                         z * block_size - depth / 2])
            colors.append([1.0, 1.0, 1.0] if (x + z) % 2 == 0 else [0, 0, 0])
    return data, colors


# Reconstructs the voxel volume from the current video frame by intersecting or unioning foreground masks across cameras.
# Supports single-camera debug, multi-camera union/intersect modes, and returns coloured voxels in visualiser coordinates.
def set_voxel_positions(width, height, depth, debug_cam=-1, debug_cams=None, debug_mode='intersect'):
    global _frame_idx

    try:
        cfgs = _ensure_cam_configs()
        bgs  = _ensure_bg_models()
        lut, vw = _ensure_lut(width, height, depth)
    except Exception:
        import traceback; traceback.print_exc()
        import random
        d, c = [], []
        for x in range(width):
            for y in range(height):
                for z in range(depth):
                    if random.randint(0, 1000) < 5:
                        d.append([x * block_size - width / 2,
                                  y * block_size,
                                  z * block_size - depth / 2])
                        c.append([x / width, z / depth, y / height])
        return d, c

    masks, frames = [], []
    for i, cid in enumerate(CAM_IDS):
        cap = cv2.VideoCapture(
            os.path.join(DATA_DIR, f'cam{cid}', 'video.avi'))
        tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fi  = _frame_idx % tot if tot > 0 else 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, fr = cap.read()
        cap.release()
        
        mean, covar = unpack_gaussian_model(fr, bgs[i])
        thresholds = optimize_thresholds(fr, mean, covar)
        if ret and bgs[i] is not None:
            frames.append(fr)
            masks.append(get_foreground_mask(fr, mean, covar, thresholds))
        else:

            frames.append(None)
            masks.append(np.zeros((480, 640), np.uint8))

    for i in range(len(CAM_IDS)):
        _last_masks[i]  = masks[i]
        _last_frames[i] = frames[i]

    _frame_idx += 1

    N = len(vw)

    if debug_cams is not None:
        cam_range = debug_cams
        print(f"[DEBUG] Cameras {[c+1 for c in cam_range]} mode={debug_mode}")
    elif 0 <= debug_cam < len(CAM_IDS):
        cam_range = [debug_cam]
        print(f"[DEBUG] Showing voxels for camera {debug_cam + 1} only")
    else:
        cam_range = list(range(len(CAM_IDS)))

    use_union = (debug_mode == 'union')

    if use_union:
        active = np.zeros(N, dtype=bool)
    else:
        active = np.ones(N, dtype=bool)

    for ci in cam_range:
        mask = masks[ci]
        h, w = mask.shape[:2]
        pr   = lut[ci]
        px   = np.round(pr[:, 0]).astype(np.int32)
        py   = np.round(pr[:, 1]).astype(np.int32)

        ok = (px >= 0) & (px < w) & (py >= 0) & (py < h) & _in_front[ci]
        fg = np.zeros(N, dtype=bool)
        vi = np.where(ok)[0]
        fg[vi] = mask[py[vi], px[vi]] > 0

        if use_union:
            active |= fg
        else:
            active &= fg

    aidx = np.where(active)[0]
    aw   = vw[aidx]
    print(f"Frame {_frame_idx - 1}: {len(aidx):,} / {N:,} active voxels")

    vx = aw[:, 0] / WORLD_SCALE
    vy = -aw[:, 2] / WORLD_SCALE
    vz = aw[:, 1] / WORLD_SCALE

    data = np.stack([vx * block_size,
                     vy * block_size,
                     vz * block_size], axis=1)

    cs = np.zeros((len(aidx), 3))
    cc = np.zeros(len(aidx))
    for ci in range(len(CAM_IDS)):
        if frames[ci] is None:
            continue
        pr = lut[ci][aidx]
        px = np.round(pr[:, 0]).astype(np.int32)
        py = np.round(pr[:, 1]).astype(np.int32)
        fh, fw = frames[ci].shape[:2]
        ok = (px >= 0) & (px < fw) & (py >= 0) & (py < fh)
        ii = np.where(ok)[0]
        if len(ii) == 0:
            continue

        # --- Per-camera Z-buffer: only the closest voxel per pixel gets color ---
        depths = _cam_depths[ci][aidx[ii]]

        # Sort descending by depth so the closest voxels (smallest depth) are
        # written last and overwrite farther ones in the buffer.
        sort_order = np.argsort(-depths)
        px_sorted = px[ii[sort_order]]
        py_sorted = py[ii[sort_order]]
        ii_sorted = ii[sort_order]

        # Scatter local-index into an image-sized buffer (last write wins = closest)
        idx_buffer = -np.ones((fh, fw), dtype=np.intp)
        idx_buffer[py_sorted, px_sorted] = ii_sorted

        # A voxel is visible from this camera iff it is still recorded in the
        # buffer at the pixel it projects to.
        visible_mask = idx_buffer[py[ii], px[ii]] == ii
        visible = ii[visible_mask]

        if len(visible) == 0:
            continue

        bgr = frames[ci][py[visible], px[visible]]
        cs[visible] += bgr[:, ::-1] / 255.0
        cc[visible] += 1

    has  = cc > 0
    cols = np.where(has[:, None], cs / np.maximum(cc[:, None], 1), 0.5)

    data   = data.tolist()
    colors = cols.tolist()

    if not data:
        data, colors = [[0, 0, 0]], [[1, 1, 1]]
    return data, colors


# Returns camera world positions converted to visualiser coordinates, along with per-camera colours.
def get_cam_positions():
    try:
        cfgs = _ensure_cam_configs()
    except Exception:
        return ([[-64, 64, 63], [63, 64, 63],
                [63, 64, -64], [-64, 64, -64]],
                [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]])

    positions = []
    for K, d, rv, tv in cfgs:
        R, _ = cv2.Rodrigues(rv)
        cw = (-R.T @ tv).flatten()
        positions.append([
            float(cw[0] / WORLD_SCALE * block_size),
            float(-cw[2] / WORLD_SCALE * block_size),
            float(cw[1] / WORLD_SCALE * block_size),
        ])
    return positions, [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]


# Builds orientation matrices for each camera in visualiser space by converting OpenCV camera axes to OpenGL convention.
def get_cam_rotation_matrices():
    try:
        cfgs = _ensure_cam_configs()
    except Exception:
        angs = [[0, 45, -45], [0, 135, -45], [0, 225, -45], [0, 315, -45]]
        rots = [glm.mat4(1)] * 4
        for c in range(4):
            rots[c] = glm.rotate(rots[c], angs[c][0] * np.pi / 180, [1, 0, 0])
            rots[c] = glm.rotate(rots[c], angs[c][1] * np.pi / 180, [0, 1, 0])
            rots[c] = glm.rotate(rots[c], angs[c][2] * np.pi / 180, [0, 0, 1])
        return rots

    rots = []
    for K, d, rv, tv in cfgs:
        R_cv, _ = cv2.Rodrigues(rv)
        right_w = R_cv.T @ np.array([1, 0, 0], dtype=np.float64)
        up_w    = R_cv.T @ np.array([0, -1, 0], dtype=np.float64)
        fwd_w   = R_cv.T @ np.array([0, 0, 1], dtype=np.float64)

        r = glm.vec3(float(right_w[0]), float(-right_w[2]), float(right_w[1]))
        u = glm.vec3(float(up_w[0]),    float(-up_w[2]),    float(up_w[1]))
        f = glm.vec3(float(fwd_w[0]),   float(-fwd_w[2]),   float(fwd_w[1]))

        mat = glm.mat4(1)
        mat[0] = glm.vec4(r, 0)
        mat[1] = glm.vec4(u, 0)
        mat[2] = glm.vec4(-f, 0)
        mat[3] = glm.vec4(0, 0, 0, 1)
        rots.append(mat)
    return rots

# endregion

# region Debug Helpers
_last_masks  = [None] * 4
_last_frames = [None] * 4


# Returns the camera's visualiser position, forward direction, last foreground mask, and last raw frame for debug overlay.
def get_debug_cam_info(cam_index):
    cfgs = _ensure_cam_configs()
    K, d, rv, tv = cfgs[cam_index]
    R_cv, _ = cv2.Rodrigues(rv)
    cw = (-R_cv.T @ tv).flatten()

    pos = glm.vec3(
        float(cw[0] / WORLD_SCALE * block_size),
        float(-cw[2] / WORLD_SCALE * block_size),
        float(cw[1] / WORLD_SCALE * block_size),
    )

    fwd_w = R_cv.T @ np.array([0, 0, 1], dtype=np.float64)
    fwd = glm.vec3(
        float(fwd_w[0]),
        float(-fwd_w[2]),
        float(fwd_w[1]),
    )
    fwd = glm.normalize(fwd)

    return pos, fwd, _last_masks[cam_index], _last_frames[cam_index]

# endregion
