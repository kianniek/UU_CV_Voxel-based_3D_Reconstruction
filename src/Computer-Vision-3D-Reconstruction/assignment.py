import glm
import numpy as np
import cv2
import cv2 as cv
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
        create_background_model_simple(cid)
    _bg_models = bg_models
    return _bg_models


# ## Task 2: Background Subtraction (Vinn)
# Focus: HSV thresholding and morphology on `background.avi` and `video.avi`.
#
#
#
# 1. Use the background.avi files to create a background model. A single frame may not give the best results.
#    Averaging frames may help to improve the background subtraction process somewhat. How to create the best
#    possible background image is up to you. You can also replace this with a more sophisticated model, e.g.
#    by modeling each pixel as a Gaussian (mixture) distribution. That will give more points.
#
# 2. Once a background model is made, you will be doing background subtraction on each frame of video.avi.
#    This means you have to find all foreground pixels. Convert the foreground frame image of a given camera
#    from BGR to HSV color space, and calculate the difference with your background model. Threshold the
#    differences for the Hue, Saturation and Value channels. See how you combine the different channels to
#    determine if a pixel is foreground (value 255) or background (value 0).
#
# 3. Setting the thresholds automatically is a choice task. Two suggestions, but there are many more viable solutions:
#    a. Make a manual segmentation of a frame into foreground and background (e.g., in Paint). Then implement a
#       function that finds the optimal thresholds by comparing the algorithm's output to the manual segmentation.
#       The XOR function might be of use here. You can look over different combinations of thresholds, and simply
#       select the combination that gave the best results.
#    b. Assume that a good segmentation has little noise (few isolated white pixels, few isolated black pixels).
#       Implement a function that tries out values and optimizes the amount of noise. Be creative in how you
#       approach this. Perhaps the functions erode and dilate can help.
#
# 4. Also think about post-processing, e.g. using erosion and dilation. Other techniques such as Blob detection
#    or Graph cuts (seam finding) could work. You don't need to implement these algorithms yourself, use the
#    OpenCV API to find the relevant functions/methods. Motivate in your report why and how you use these.
#
#
# ## Sources
#
# - https://en.wikipedia.org/wiki/Foreground_detection
# - https://en.wikipedia.org/wiki/Moving_average
# - https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
# - https://github.com/pransen/ComputerVisionAlgorithms/blob/master/GaussianMixtureModelling/README.md

FRAME_RATE = 30  # Assuming a frame rate of 30 FPS for the videos
bg_models = []

#region Helper Functions for retrieving video material
def get_background_video(cam_id):
    return cv.VideoCapture(os.path.join(DATA_DIR, f'cam{cam_id}', 'background.avi'))

def get_video(cam_id):
    return cv.VideoCapture(os.path.join(DATA_DIR, f'cam{cam_id}', 'video.avi'))
#endregion

# region simple background subtraction method using frame averaging
def create_background_model_simple(cam_id):
    """"Creates a background model for a given camera by averaging frames from the background video."""
    
    video = get_background_video(cam_id)
    
    ret, frame = video.read()
    if not ret:
        print(f"Error reading video for camera {cam_id}")
        return None
    base = frame.astype(np.float32)
    
    # extract frames, average them, and create a background model per camera
    while True:
        ret, frame = video.read()
        if not ret:
            break;
        
        cv.accumulateWeighted(frame, base, 0.01)
    
    bg_models.append(cv.convertScaleAbs(base))
    
    video.release()

# endregion

# region a bit more advanced bacgkround subtraction using Mixture of Gaussians 
def create_background_model_mog(cam_id):
    frame_id = 0
    count = 0
    
    model = []
    while True:
        video = get_background_video(cam_id)
        ret, frame = video.read()

        # 1. Extract frames from the video.
        # 2. Stack the frames in an array where the final array dimensions will be (num_frames, image_width, image_height, num_channels)
        if ret:
            # grab a frame every 30 frames
            model.append([frame_id, frame.shape[0], frame.shape[1], frame.shape[2]])
            count += FRAME_RATE
            video.set(cv.CAP_PROP_POS_FRAMES, count)
        else:
            video.release()
            break
        
    # 3. Initialize a dummy background image of the same size as that of the individual frames.
    base_bg = np.zeros((model[0][1], model[0][2], model[0][3]), dtype=np.float32)
    
    # 4. For each point characterized by the x-coordinate, the y-coordinate and the channel, 
    # model the intensity value across all the frames as a mixture of two Gaussians.
    # TODO: https://www.youtube.com/watch?v=0nz8JMyFF14 (guy explaining mixture of gaussians in a simple way)
    
    # so for every x, y, and channel of a pixel? we're going across all pixels?
    
    # 5. Once modelled, initialize the intensity value at the corresponding location in the dummy background image with the mean of the most weighted cluster. 
    #       Intensity == Value in HSV?   
    # The most weighted cluster will be the one coming from the background whereas owing to the dynamically changing and sparse nature of the foreground, the other cluster will be weighted less.
    
    # 6. Finally, the background image will contain the intensity values corresponding to the static background.

    bg_models.append(model)

#create a foreground mask per camera using the background model made earlier
def get_foreground_mask_simple(frame, bg_model, thresholds = [15, 15, 15]):
    # TODO: BGR to HSV -> Diff -> Threshold -> Erode/Dilate

    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    bg_model_hsv = cv.cvtColor(bg_model, cv.COLOR_BGR2HSV)

    # Compute absolute difference between current frame and background model in HSV space
    frame_diff = cv.absdiff(frame_hsv, bg_model_hsv)
    
    # split frame into H, S, V channels and apply thresholding to S and V channels
    # for H channel, we can ignore it for now (seems to be less susceptible to change if you see output)
    h,s,v, = cv.split(frame_diff)
    s_mask = cv.threshold(s, thresholds[1], 255, cv.THRESH_BINARY)[1]
    v_mask = cv.threshold(v, thresholds[2], 255, cv.THRESH_BINARY)[1]
    mask = cv.merge((h, s_mask, v_mask))
    
    mask = cv.cvtColor(cv.cvtColor(mask, cv.COLOR_HSV2BGR), cv.COLOR_BGR2GRAY)
    mask = cv.threshold(mask, 15, 255, cv.THRESH_BINARY)[1]
    
    # Morphological operations to clean up the mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    return mask

# CHOICE 2: Automated thresholding function
def optimize_thresholds(frame, manual_segmentation):
    pass
# endregion


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

        if ret and bgs[i] is not None:
            frames.append(fr)
            masks.append(get_foreground_mask_simple(fr, bgs[i]))
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
