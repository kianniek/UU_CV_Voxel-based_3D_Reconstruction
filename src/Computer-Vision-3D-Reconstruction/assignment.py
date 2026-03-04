import glm
import numpy as np
import cv2
import cv2 as cv
import os
import xml.etree.ElementTree as ET

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

CHECKERBOARD_SIZE = (6, 8)  # Defined in checkerboard.xml
SQUARE_SIZE = 25             # mm

def get_config_path(cam_id):
    return os.path.join(DATA_DIR, f'cam{cam_id}', 'config.xml')

# endregion

# region Task 1: Calibration – Automatic Chessboard Corner Detection

def _preprocess_variants(gray):
    """Yield (label, image) pairs with increasingly aggressive preprocessing."""
    yield "original", gray

    yield "blur_5", cv2.GaussianBlur(gray, (5, 5), 0)
    yield "blur_7", cv2.GaussianBlur(gray, (7, 7), 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    yield "clahe", clahe.apply(gray)
    yield "clahe_blur", cv2.GaussianBlur(clahe.apply(gray), (5, 5), 0)

    eq = cv2.equalizeHist(gray)
    yield "histeq", eq
    yield "histeq_blur", cv2.GaussianBlur(eq, (5, 5), 0)

    yield "bilateral", cv2.bilateralFilter(gray, 9, 75, 75)

    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    sharp = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    yield "sharpened", sharp


def auto_detect_chessboard_corners(cam_id, cols, rows):
    """Fully-automatic chessboard corner detection from checkerboard.avi.

    Tries multiple frames and preprocessing strategies until the board is
    found.  Returns:
        corners2d  – (rows*cols, 1, 2) refined inner corners
        handles    – (4, 2) the four extreme corners [TL, TR, BR, BL]
    """
    video_path = os.path.join(DATA_DIR, f'cam{cam_id}', 'checkerboard.avi')
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pattern_size = (cols, rows)

    cb_flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
                cv2.CALIB_CB_NORMALIZE_IMAGE |
                cv2.CALIB_CB_FAST_CHECK)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                100, 0.001)

    sample_indices = sorted(set([
        frame_count // 2,
        0,
        frame_count - 1,
        frame_count // 4,
        3 * frame_count // 4,
        frame_count // 3,
        2 * frame_count // 3,
    ]))

    best_corners = None
    best_gray = None
    best_label = ""

    for fidx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for label, img in _preprocess_variants(gray):
            found, corners = cv2.findChessboardCorners(img, pattern_size, cb_flags)
            if found:
                refined = cv2.cornerSubPix(gray, corners, (11, 11),
                                           (-1, -1), criteria)
                best_corners = refined
                best_gray = gray
                best_label = f"findChessboardCorners/{label}/frame{fidx}"
                break

        if best_corners is not None:
            break

        try:
            found, corners = cv2.findChessboardCornersSB(gray, pattern_size)
            if found:
                best_corners = corners
                best_gray = gray
                best_label = f"findChessboardCornersSB/frame{fidx}"
        except AttributeError:
            pass

        if best_corners is not None:
            break

    cap.release()

    if best_corners is None:
        raise RuntimeError(
            f"Cam {cam_id}: automatic chessboard detection failed on all "
            f"{len(sample_indices)} sampled frames with all preprocessing "
            f"strategies.  Consider falling back to manual calibration.")

    print(f"  Cam {cam_id}: auto-detected via {best_label}")

    c = best_corners.reshape(-1, 2)
    idx_tl = 0
    idx_tr = cols - 1
    idx_br = rows * cols - 1
    idx_bl = (rows - 1) * cols

    handles = np.array([
        c[idx_tl],
        c[idx_tr],
        c[idx_br],
        c[idx_bl],
    ], dtype=np.float32)

    return best_corners, handles


# Projects a grid onto the image plane using a perspective transform derived from the corners
def project_grid_perspective(four_corners, cols, rows):
    src_ideal = np.array([
        [0, 0], [cols-1, 0], [cols-1, rows-1], [0, rows-1]
    ], dtype=np.float32)
    dst_current = np.array(four_corners, dtype=np.float32)
    H_matrix = cv2.getPerspectiveTransform(src_ideal, dst_current)
    grid_x, grid_y = np.meshgrid(np.arange(cols), np.arange(rows))
    ideal_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T.astype(np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(ideal_points, H_matrix)


# Warps the image region to a flat view to refine corner detection, then projects points back
def refine_corners_via_warping(image_gray, four_corners, cols, rows):
    w_top = np.linalg.norm(four_corners[0] - four_corners[1])
    w_bot = np.linalg.norm(four_corners[3] - four_corners[2])
    h_left = np.linalg.norm(four_corners[0] - four_corners[3])
    h_right = np.linalg.norm(four_corners[1] - four_corners[2])
    dst_width = int(max(w_top, w_bot))
    dst_height = int(max(h_left, h_right))
    padding = 50
    dst_corners = np.array([
        [padding, padding],
        [padding + dst_width, padding],
        [padding + dst_width, padding + dst_height],
        [padding, padding + dst_height]
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(np.array(four_corners, dtype=np.float32), dst_corners)
    M_inv = np.linalg.inv(M)
    warped_image = cv2.warpPerspective(image_gray, M, (dst_width + 2*padding, dst_height + 2*padding))
    grid_x, grid_y = np.meshgrid(
        np.linspace(padding, padding + dst_width, cols),
        np.linspace(padding, padding + dst_height, rows)
    )
    warped_guesses = np.vstack([grid_x.ravel(), grid_y.ravel()]).T.astype(np.float32)
    warped_guesses = np.ascontiguousarray(warped_guesses).reshape(-1, 1, 2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    refined_warped = cv2.cornerSubPix(warped_image, warped_guesses, (5, 5), (-1, -1), criteria)
    return cv2.perspectiveTransform(refined_warped, M_inv)


# Interactive GUI with Magnifying Glass to allow users to move 4 corners (manual fallback)
def select_corners_interface(image_gray, cols, rows, initial_handles=None):
    CLICK_SENSITIVITY = 20
    HANDLE_RADIUS = 1
    GRID_RADIUS = 1
    height, width = image_gray.shape
    if initial_handles is not None:
        handles = [np.array(pt, dtype=np.float32) for pt in initial_handles]
    else:
        margin_horizontal, margin_vertical = width // 4, height // 4
        handles = [
            np.array([margin_horizontal, margin_vertical], dtype=np.float32),
            np.array([width - margin_horizontal, margin_vertical], dtype=np.float32),
            np.array([width - margin_horizontal, height - margin_vertical], dtype=np.float32),
            np.array([margin_horizontal, height - margin_vertical], dtype=np.float32)
        ]
    state = {
        "drag_index": None,
        "mouse_pos": (width // 2, height // 2),
        "handles": handles
    }
    window_name = "Manual Calibration - Drag corners. Press 'c' to confirm"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1080, int(1080 * (height/width)))

    def mouse_callback(event, x, y, flags, param):
        state["mouse_pos"] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            for idx, pt in enumerate(state["handles"]):
                if np.linalg.norm(pt - [x, y]) < CLICK_SENSITIVITY:
                    state["drag_index"] = idx
                    break
        elif event == cv2.EVENT_MOUSEMOVE and state["drag_index"] is not None:
            state["handles"][state["drag_index"]] = np.array([x, y], dtype=np.float32)
        elif event == cv2.EVENT_LBUTTONUP:
            if state["drag_index"] is not None:
                corner = np.array([[[x, y]]], dtype=np.float32)
                corner = np.ascontiguousarray(corner)
                state["handles"][state["drag_index"]] = corner[0, 0]
                state["drag_index"] = None

    cv2.setMouseCallback(window_name, mouse_callback)
    display_color = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 255, 255)]
    while True:
        temp_img = display_color.copy()
        corner_points = np.array(state["handles"], np.int32).reshape((-1, 1, 2))
        cv2.polylines(temp_img, [corner_points], True, (255, 255, 0), 1)
        grid_points = project_grid_perspective(state["handles"], cols, rows)
        if grid_points is not None:
            for point in grid_points:
                cv2.circle(temp_img, tuple(point[0].astype(int)), GRID_RADIUS, (0, 255, 0), -1)
        for idx, point in enumerate(state["handles"]):
            cv2.circle(temp_img, tuple(point.astype(int)), HANDLE_RADIUS, colors[idx], -1)
        mouse_x = np.clip(state["mouse_pos"][0], 0, width - 1)
        mouse_y = np.clip(state["mouse_pos"][1], 0, height - 1)
        pad_size = 30
        pad_img = cv2.copyMakeBorder(temp_img, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REPLICATE)
        cx, cy = mouse_x + pad_size, mouse_y + pad_size
        patch = pad_img[cy-pad_size:cy+pad_size, cx-pad_size:cx+pad_size]
        zoom_size = pad_size * 4
        patch_zoomed = cv2.resize(patch, (zoom_size, zoom_size), interpolation=cv2.INTER_NEAREST)
        cv2.line(patch_zoomed, (zoom_size//2, 0), (zoom_size//2, zoom_size), (255, 255, 255), 1)
        cv2.line(patch_zoomed, (0, zoom_size//2), (zoom_size, zoom_size//2), (255, 255, 255), 1)
        cv2.rectangle(patch_zoomed, (0,0), (zoom_size-1, zoom_size-1), (255, 255, 255), 2)
        if mouse_x > width - zoom_size - 20 and mouse_y < zoom_size + 20:
            temp_img[10:10+zoom_size, 10:10+zoom_size] = patch_zoomed
        else:
            temp_img[10:10+zoom_size, width-zoom_size-10:width-10] = patch_zoomed
        cv2.imshow(window_name, temp_img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            try:
                refined_corner_points = refine_corners_via_warping(image_gray, state["handles"], cols, rows)
                cv2.destroyWindow(window_name)
                return refined_corner_points, np.array(state["handles"])
            except Exception as e:
                print(f"Refinement failed: {e}")


def calibrate_intrinsics(cam_id):
    """Calibrate camera intrinsics using intrinsics.avi and checkerboard.xml."""
    xml_path = os.path.join(DATA_DIR, 'checkerboard.xml')
    tree = ET.parse(xml_path)
    root = tree.getroot()
    cols = int(root.find("CheckerBoardWidth").text)
    rows = int(root.find("CheckerBoardHeight").text)
    square_size = float(root.find("CheckerBoardSquareSize").text)

    object_points = np.zeros((rows * cols, 3), np.float32)
    object_points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    object_points *= square_size

    video_path = os.path.join(DATA_DIR, f'cam{cam_id}', 'intrinsics.avi')
    cap = cv2.VideoCapture(video_path)
    objpoints = []
    imgpoints = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, frame_count // 10)
    for i in range(0, frame_count, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
        if found:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            objpoints.append(object_points)
            imgpoints.append(corners2)
    cap.release()
    if len(objpoints) < 3:
        raise RuntimeError("Not enough valid frames for calibration.")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx, dist


def calibrate_extrinsics(cam_id, mtx, dist):
    """Calibrate camera extrinsics – fully automatic with manual fallback."""
    xml_path = os.path.join(DATA_DIR, 'checkerboard.xml')
    tree = ET.parse(xml_path)
    root = tree.getroot()
    cols = int(root.find("CheckerBoardWidth").text)
    rows = int(root.find("CheckerBoardHeight").text)
    square_size = float(root.find("CheckerBoardSquareSize").text)

    object_points = np.zeros((rows * cols, 3), np.float32)
    object_points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    object_points *= square_size

    print(f"--- Extrinsic Calibration: Camera {cam_id} ---")

    try:
        corners2d, handles = auto_detect_chessboard_corners(cam_id, cols, rows)
        print(f"  Cam {cam_id}: automatic detection succeeded.")
    except RuntimeError as e:
        print(f"  Cam {cam_id}: auto-detection failed ({e}), falling back to manual...")
        initial_handles = None
        config_path = get_config_path(cam_id)
        if os.path.exists(config_path):
            fs_read = cv2.FileStorage(config_path, cv2.FILE_STORAGE_READ)
            corner_node = fs_read.getNode("manual_corners")
            if not corner_node.empty():
                initial_handles = corner_node.mat()
            fs_read.release()

        video_path = os.path.join(DATA_DIR, f'cam{cam_id}', 'checkerboard.avi')
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Could not read checkerboard frame for Cam {cam_id}.")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners2d, handles = select_corners_interface(gray, cols, rows, initial_handles)

    ret, rvec, tvec = cv2.solvePnP(object_points, corners2d, mtx, dist)
    return rvec, tvec, handles


def save_all_params(cam_id, mtx, dist, rvec, tvec, manual_corners):
    """Save all calibration parameters to an XML file for future use."""
    fs = cv2.FileStorage(get_config_path(cam_id), cv2.FILE_STORAGE_WRITE)
    fs.write("camera_matrix", mtx)
    fs.write("distortion_coefficients", dist)
    fs.write("rotation_vector", rvec)
    fs.write("translation_vector", tvec)
    fs.write("manual_corners", manual_corners)
    fs.release()


def visualize_checkerboard_start(cam_id, mtx, dist, rvec, tvec):
    """Project the world origin axes onto a checkerboard frame for visual verification."""
    xml_path = os.path.join(DATA_DIR, 'checkerboard.xml')
    tree = ET.parse(xml_path)
    root = tree.getroot()
    square_size = float(root.find("CheckerBoardSquareSize").text)
    axis_length = square_size * 3
    axis_points = np.float32([
        [0, 0, 0],
        [axis_length, 0, 0],
        [0, axis_length, 0],
        [0, 0, -axis_length]
    ])
    axis_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, mtx, dist)
    imgpts = imgpts.reshape(-1, 2).astype(int)

    video_path = os.path.join(DATA_DIR, f'cam{cam_id}', 'checkerboard.avi')
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 5)
    ret, frame = cap.read()
    cap.release()

    if ret:
        origin = tuple(imgpts[0])
        pt_x = tuple(imgpts[1])
        pt_y = tuple(imgpts[2])
        pt_z = tuple(imgpts[3])
        cv2.line(frame, origin, pt_x, axis_colors[0], 2)
        cv2.line(frame, origin, pt_y, axis_colors[1], 2)
        cv2.line(frame, origin, pt_z, axis_colors[2], 2)
        cv2.circle(frame, origin, 3, (255, 255, 255), -1)
        cv2.imshow(f"Camera {cam_id} Axes", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"Error: Could not load frame for Camera {cam_id}")


def verify_origin_alignment():
    """Show all 4 cameras side-by-side with projected world origin + axes."""
    xml_path = os.path.join(DATA_DIR, 'checkerboard.xml')
    tree = ET.parse(xml_path)
    root = tree.getroot()
    square_size = float(root.find("CheckerBoardSquareSize").text)

    axis_length = square_size * 3
    axis_pts_3d = np.float32([
        [0, 0, 0],
        [axis_length, 0, 0],
        [0, axis_length, 0],
        [0, 0, -axis_length],
    ])

    panels = []
    for cid in CAM_IDS:
        fs = cv2.FileStorage(get_config_path(cid), cv2.FILE_STORAGE_READ)
        K    = fs.getNode("camera_matrix").mat()
        dist = fs.getNode("distortion_coefficients").mat()
        rv   = fs.getNode("rotation_vector").mat()
        tv   = fs.getNode("translation_vector").mat()
        fs.release()

        cap = cv2.VideoCapture(os.path.join(DATA_DIR, f'cam{cid}', 'checkerboard.avi'))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, n_frames // 2)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            frame = np.zeros((480, 640, 3), np.uint8)

        imgpts, _ = cv2.projectPoints(axis_pts_3d, rv, tv, K, dist)
        imgpts = imgpts.reshape(-1, 2).astype(int)
        o, px, py, pz = [tuple(p) for p in imgpts]

        cv2.line(frame, o, px, (0, 0, 255), 3)
        cv2.line(frame, o, py, (0, 255, 0), 3)
        cv2.line(frame, o, pz, (255, 0, 0), 3)
        cv2.circle(frame, o, 5, (255, 255, 255), -1)

        cols = int(root.find("CheckerBoardWidth").text)
        rows = int(root.find("CheckerBoardHeight").text)
        board_corners_3d = np.float32([
            [0, 0, 0],
            [(cols - 1) * square_size, 0, 0],
            [(cols - 1) * square_size, (rows - 1) * square_size, 0],
            [0, (rows - 1) * square_size, 0],
        ])
        bc_px, _ = cv2.projectPoints(board_corners_3d, rv, tv, K, dist)
        bc_px = bc_px.reshape(-1, 2).astype(int)
        for j in range(4):
            cv2.line(frame, tuple(bc_px[j]), tuple(bc_px[(j + 1) % 4]),
                     (0, 255, 255), 1)

        cv2.putText(frame, f"Cam {cid}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "origin = white dot", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        panels.append(frame)

    h, w = panels[0].shape[:2]
    for i in range(len(panels)):
        panels[i] = cv2.resize(panels[i], (w, h))
    top = np.hstack(panels[:2])
    bot = np.hstack(panels[2:])
    grid = np.vstack([top, bot])

    cv2.imshow("Origin Alignment Check", grid)
    print("CHECK: The white dot + RGB axes must sit on the SAME physical")
    print("corner of the checkerboard in all 4 views.  Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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

# region Task 3: Standalone Voxel Reconstruction (notebook pipeline)

def load_camera_config(cam_id):
    """Load camera calibration parameters from saved config.xml."""
    config_path = get_config_path(cam_id)
    fs = cv2.FileStorage(config_path, cv2.FILE_STORAGE_READ)
    mtx  = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("distortion_coefficients").mat()
    rvec = fs.getNode("rotation_vector").mat()
    tvec = fs.getNode("translation_vector").mat()
    fs.release()
    return mtx, dist, rvec, tvec


def build_lookup_table(cam_configs, grid_width=128, grid_height=64, grid_depth=128):
    """Build lookup table: project each 3D voxel position to 2D for each camera.

    Returns
    -------
    voxel_positions : ndarray  (N, 3) – world coordinates (mm)
    lookup_table    : dict  cam_index -> (N, 2) pixel coords
    in_front        : dict  cam_index -> bool array (N,)
    """
    step = VOXEL_STEP
    half_w = (grid_width  / 2) * step
    half_d = (grid_depth  / 2) * step
    h_max  = grid_height * step

    xs = np.arange(-half_w, half_w, step, dtype=np.float64)
    ys = np.arange(-half_d, half_d, step, dtype=np.float64)
    zs = np.arange(-h_max,  0,      step, dtype=np.float64)

    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing='ij')
    voxel_positions = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

    N = len(voxel_positions)
    print(f"Building LUT: {len(xs)}x{len(ys)}x{len(zs)} = {N:,} voxels "
          f"(step {step} mm)")

    lookup_table = {}
    in_front     = {}
    for i, (mtx, dist, rvec, tvec) in enumerate(cam_configs):
        proj, _ = cv2.projectPoints(voxel_positions, rvec, tvec, mtx, dist)
        lookup_table[i] = proj.reshape(-1, 2)

        R, _ = cv2.Rodrigues(rvec)
        P_cam = (voxel_positions @ R.T) + tvec.T
        in_front[i] = P_cam[:, 2] > 0

        print(f"  Cam {i+1}: {np.sum(in_front[i]):,}/{N:,} in front")

    return voxel_positions, lookup_table, in_front


def reconstruct_voxels(masks, lookup_table, voxel_positions, in_front, frames=None):
    """Reconstruct voxels by intersecting foreground across all cameras."""
    N = len(voxel_positions)
    active = np.ones(N, dtype=bool)

    for ci in range(len(masks)):
        mask = masks[ci]
        h, w = mask.shape[:2]
        pr   = lookup_table[ci]
        px   = np.round(pr[:, 0]).astype(np.int32)
        py   = np.round(pr[:, 1]).astype(np.int32)

        ok = (px >= 0) & (px < w) & (py >= 0) & (py < h) & in_front[ci]
        fg = np.zeros(N, dtype=bool)
        vi = np.where(ok)[0]
        fg[vi] = mask[py[vi], px[vi]] > 0
        active &= fg

    aidx = np.where(active)[0]
    aw   = voxel_positions[aidx]
    print(f"Active voxels: {len(aidx):,} / {N:,}")

    vx =  aw[:, 0] / VOXEL_STEP
    vy = -aw[:, 2] / VOXEL_STEP
    vz =  aw[:, 1] / VOXEL_STEP
    data = np.stack([vx, vy, vz], axis=1).tolist()

    if frames is not None and any(f is not None for f in frames):
        cs = np.zeros((len(aidx), 3))
        cc = np.zeros(len(aidx))
        for ci in range(len(frames)):
            if frames[ci] is None:
                continue
            pr = lookup_table[ci][aidx]
            px = np.round(pr[:, 0]).astype(np.int32)
            py = np.round(pr[:, 1]).astype(np.int32)
            fh, fw = frames[ci].shape[:2]
            ok = (px >= 0) & (px < fw) & (py >= 0) & (py < fh)
            ii = np.where(ok)[0]
            if len(ii):
                bgr = frames[ci][py[ii], px[ii]]
                cs[ii] += bgr[:, ::-1] / 255.0
                cc[ii] += 1
        has  = cc > 0
        cols = np.where(has[:, None], cs / np.maximum(cc[:, None], 1), 0.5)
        colors = cols.tolist()
    else:
        colors = [[0.5, vy_i / 64.0, 0.5] for vy_i in vy]

    if not data:
        data, colors = [[0, 0, 0]], [[1, 1, 1]]
    return data, colors

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


# region Main Pipeline

def run_pipeline():
    """Run the full calibration → background subtraction → voxel reconstruction pipeline."""

    # 1. Run Calibration
    print("=== Step 1: Camera Calibration ===")
    for cam_id in CAM_IDS:
        mtx, dist = calibrate_intrinsics(cam_id)
        rvec, tvec, handles = calibrate_extrinsics(cam_id, mtx, dist)
        save_all_params(cam_id, mtx, dist, rvec, tvec, handles)
        print(f"Visualizing Origin for Camera {cam_id}...")
        visualize_checkerboard_start(cam_id, mtx, dist, rvec, tvec)

    # 1b. Verify all cameras share the same world origin
    print("\n=== Verifying Origin Alignment ===")
    verify_origin_alignment()

    # 2. Ensure BG Models are available
    print("\n=== Step 2: Background Subtraction ===")
    if len(bg_models) < len(CAM_IDS):
        for cid in CAM_IDS:
            if cid - 1 >= len(bg_models) or bg_models[cid - 1] is None:
                create_background_model_simple(cid)

    # 3. Show foreground masks for each camera in sequence
    exit_all = False
    for cam_id in CAM_IDS:
        if exit_all:
            break
        video = get_video(cam_id)
        window_name = f'Camera {cam_id} - Foreground Mask (q: next, x/ESC: exit)'
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        while True:
            ret, frame = video.read()
            if not ret:
                break
            mask = get_foreground_mask_simple(frame, bg_models[cam_id - 1])
            cv.imshow(window_name, mask)
            key = cv.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            if key == ord('x') or key == 27:
                exit_all = True
                break
        video.release()
        cv.destroyWindow(window_name)

    # 4. Build Voxel Lookup Table
    print("\n=== Step 3: Building Voxel Lookup Table ===")
    cam_configs = []
    for cam_id in CAM_IDS:
        cam_configs.append(load_camera_config(cam_id))
    voxel_positions, lookup_table, in_front = build_lookup_table(cam_configs)

    # 5. Reconstruct Voxels from first frame
    print("\n=== Step 4: Voxel Reconstruction ===")
    masks_for_recon = []
    frames_for_color = []
    for i, cam_id in enumerate(CAM_IDS):
        video = get_video(cam_id)
        ret, frame = video.read()
        video.release()
        if ret:
            frames_for_color.append(frame)
            masks_for_recon.append(
                get_foreground_mask_simple(frame, bg_models[cam_id - 1]))
        else:
            frames_for_color.append(None)
            masks_for_recon.append(np.zeros((480, 640), np.uint8))

    voxel_data, voxel_colors = reconstruct_voxels(
        masks_for_recon, lookup_table, voxel_positions, in_front,
        frames=frames_for_color)

    # Save results for use by the 3D visualiser
    np.savez(os.path.join(DATA_DIR, 'voxel_reconstruction.npz'),
             data=np.array(voxel_data, dtype=np.float32),
             colors=np.array(voxel_colors, dtype=np.float32))
    print(f"Saved {len(voxel_data)} voxels to data/voxel_reconstruction.npz")
    print("\nPipeline complete.")


if __name__ == '__main__':
    run_pipeline()

# endregion
