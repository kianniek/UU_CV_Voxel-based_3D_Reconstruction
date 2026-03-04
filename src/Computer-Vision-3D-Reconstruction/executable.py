import glm
import glfw
from engine.base.program import get_linked_program
from engine.renderable.model import Model
from engine.buffer.texture import *
from engine.buffer.hdrbuffer import HDRBuffer
from engine.buffer.blurbuffer import BlurBuffer
from engine.effect.bloom import Bloom
from assignment import set_voxel_positions, generate_grid, get_cam_positions, get_cam_rotation_matrices, get_debug_cam_info
from engine.camera import Camera
from engine.config import config
import cv2
import math
import numpy as np

cube, hdrbuffer, blurbuffer, lastPosX, lastPosY = None, None, None, None, None
firstTime = True
window_width, window_height = config['window_width'], config['window_height']
camera = Camera(glm.vec3(0, 100, 0), pitch=-90, yaw=0, speed=40)

_debug_cam_index = -1
_debug_overlay_window = None

_seq_active = False
_seq_cam_index = 0
_seq_next_time = 0.0
_seq_mode = 'union'
_SEQ_DELAY = 5.0

_playback_active = False
_playback_next_time = 0.0
_PLAYBACK_FPS = 30.0
_g_pressed_once = False

# ── Orbit camera mode ────────────────────────────────────────────────
_orbit_active = False
_orbit_angle = 0.0            # current horizontal angle (radians)
_orbit_radius = 150.0         # distance from centre
_orbit_speed = 0.5            # radians / second (current, lerped)
_orbit_speed_target = 0.5     # target after key press
_orbit_pitch = 30.0           # elevation in degrees  (current, lerped)
_orbit_pitch_target = 30.0    # target after key press
_orbit_center = glm.vec3(0, 0, 0)
_ORBIT_SPEED_STEP = 0.25      # increment per arrow press
_ORBIT_PITCH_STEP = 10.0      # degrees per arrow press
_ORBIT_LERP_RATE = 4.0        # how fast values approach target (per sec)
_last_voxel_positions = None   # cached for orbit centre


def draw_objs(obj, program, perspective, light_pos, texture, normal, specular, depth):
    program.use()
    program.setMat4('viewProject', perspective * camera.get_view_matrix())
    program.setVec3('viewPos', camera.position)
    program.setVec3('light_pos', light_pos)

    glActiveTexture(GL_TEXTURE1)
    program.setInt('mat.diffuseMap', 1)
    texture.bind()

    glActiveTexture(GL_TEXTURE2)
    program.setInt('mat.normalMap', 2)
    normal.bind()

    glActiveTexture(GL_TEXTURE3)
    program.setInt('mat.specularMap', 3)
    specular.bind()

    glActiveTexture(GL_TEXTURE4)
    program.setInt('mat.depthMap', 4)
    depth.bind()
    program.setFloat('mat.shininess', 128)
    program.setFloat('mat.heightScale', 0.12)

    obj.draw_multiple(program)


def main():
    global hdrbuffer, blurbuffer, cube, window_width, window_height

    if not glfw.init():
        print('Failed to initialize GLFW.')
        return

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
    glfw.window_hint(glfw.SAMPLES, config['sampling_level'])

    if config['fullscreen']:
        mode = glfw.get_video_mode(glfw.get_primary_monitor())
        window_width, window_height = mode.size.window_width, mode.size.window_height
        window = glfw.create_window(mode.size.window_width,
                                    mode.size.window_height,
                                    config['app_name'],
                                    glfw.get_primary_monitor(),
                                    None)
    else:
        window = glfw.create_window(window_width, window_height, config['app_name'], None, None)
    if not window:
        print('Failed to create GLFW Window.')
        glfw.terminate()
        return

    glfw.make_context_current(window)
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
    glfw.set_framebuffer_size_callback(window, resize_callback)
    glfw.set_cursor_pos_callback(window, mouse_move)
    glfw.set_key_callback(window, key_callback)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_MULTISAMPLE)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)

    program = get_linked_program('resources/shaders/vert.vs', 'resources/shaders/frag.fs')
    depth_program = get_linked_program('resources/shaders/shadow_depth.vs', 'resources/shaders/shadow_depth.fs')
    blur_program = get_linked_program('resources/shaders/blur.vs', 'resources/shaders/blur.fs')
    hdr_program = get_linked_program('resources/shaders/hdr.vs', 'resources/shaders/hdr.fs')

    blur_program.use()
    blur_program.setInt('image', 0)

    hdr_program.use()
    hdr_program.setInt('sceneMap', 0)
    hdr_program.setInt('bloomMap', 1)

    window_width_px, window_height_px = glfw.get_framebuffer_size(window)

    hdrbuffer = HDRBuffer()
    hdrbuffer.create(window_width_px, window_height_px)
    blurbuffer = BlurBuffer()
    blurbuffer.create(window_width_px, window_height_px)

    bloom = Bloom(hdrbuffer, hdr_program, blurbuffer, blur_program)

    light_pos = glm.vec3(0.5, 0.5, 0.5)
    perspective = glm.perspective(45, window_width / window_height, config['near_plane'], config['far_plane'])

    cam_rot_matrices = get_cam_rotation_matrices()
    cam_shapes = [Model('resources/models/camera.json', cam_rot_matrices[c]) for c in range(4)]
    square = Model('resources/models/square.json')
    cube = Model('resources/models/cube.json')
    texture = load_texture_2d('resources/textures/diffuse.jpg')
    texture_grid = load_texture_2d('resources/textures/diffuse_grid.jpg')
    normal = load_texture_2d('resources/textures/normal.jpg')
    normal_grid = load_texture_2d('resources/textures/normal_grid.jpg')
    specular = load_texture_2d('resources/textures/specular.jpg')
    specular_grid = load_texture_2d('resources/textures/specular_grid.jpg')
    depth = load_texture_2d('resources/textures/depth.jpg')
    depth_grid = load_texture_2d('resources/textures/depth_grid.jpg')

    grid_positions, grid_colors = generate_grid(config['world_width'], config['world_width'])
    square.set_multiple_positions(grid_positions, grid_colors)

    cam_positions, cam_colors = get_cam_positions()
    for c, cam_pos in enumerate(cam_positions):
        cam_shapes[c].set_multiple_positions([cam_pos], [cam_colors[c]])

    last_time = glfw.get_time()
    while not glfw.window_should_close(window):
        if config['debug_mode']:
            print(glGetError())

        current_time = glfw.get_time()
        delta_time = current_time - last_time
        last_time = current_time

        move_input(window, delta_time)
        _orbit_update(delta_time)

        if _seq_active and current_time >= _seq_next_time:
            _seq_advance()

        if _playback_active and current_time >= _playback_next_time:
            _playback_tick()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0.1, 0.2, 0.8, 1)

        square.draw_multiple(depth_program)
        cube.draw_multiple(depth_program)
        for cam in cam_shapes:
            cam.draw_multiple(depth_program)

        hdrbuffer.bind()

        window_width_px, window_height_px = glfw.get_framebuffer_size(window)
        glViewport(0, 0, window_width_px, window_height_px)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        draw_objs(square, program, perspective, light_pos, texture_grid, normal_grid, specular_grid, depth_grid)
        draw_objs(cube, program, perspective, light_pos, texture, normal, specular, depth)
        for cam in cam_shapes:
            draw_objs(cam, program, perspective, light_pos, texture_grid, normal_grid, specular_grid, depth_grid)

        hdrbuffer.unbind()
        hdrbuffer.finalize()

        bloom.draw_processed_scene()

        glfw.poll_events()
        glfw.swap_buffers(window)

    glfw.terminate()


def resize_callback(window, w, h):
    if h > 0:
        global window_width, window_height, hdrbuffer, blurbuffer
        window_width, window_height = w, h
        glm.perspective(45, window_width / window_height, config['near_plane'], config['far_plane'])
        window_width_px, window_height_px = glfw.get_framebuffer_size(window)
        hdrbuffer.delete()
        hdrbuffer.create(window_width_px, window_height_px)
        blurbuffer.delete()
        blurbuffer.create(window_width_px, window_height_px)


def _snap_camera_to(cam_idx):
    """Snap the free-roam camera to the position/orientation of cam_idx (0-3)."""
    global camera, _debug_cam_index, _debug_overlay_window

    pos, fwd, mask, frame = get_debug_cam_info(cam_idx)
    camera.position = pos

    camera.yaw = math.degrees(math.atan2(fwd.z, fwd.x))
    camera.pitch = math.degrees(math.asin(max(-1.0, min(1.0, fwd.y))))
    camera.update_vectors()

    _debug_cam_index = cam_idx

    win_name = f'Debug – Camera {cam_idx + 1} mask'
    if _debug_overlay_window and _debug_overlay_window != win_name:
        cv2.destroyWindow(_debug_overlay_window)
    _debug_overlay_window = win_name

    if mask is not None and frame is not None:
        overlay = frame.copy()
        green = overlay.copy()
        green[:] = (0, 255, 0)
        mask_3ch = cv2.merge([mask, mask, mask])
        overlay = np.where(mask_3ch > 0, cv2.addWeighted(overlay, 0.5, green, 0.5, 0), overlay)
        cv2.imshow(win_name, overlay)
        cv2.waitKey(1)
    elif mask is not None:
        cv2.imshow(win_name, mask)
        cv2.waitKey(1)
    else:
        print(f'No mask available yet – press G first to run a reconstruction.')


def _close_debug_overlay():
    global _debug_cam_index, _debug_overlay_window
    if _debug_overlay_window:
        cv2.destroyWindow(_debug_overlay_window)
        _debug_overlay_window = None
    _debug_cam_index = -1


def _show_all_masks():
    """Display all 4 camera masks side-by-side in a 2x2 mosaic."""
    from assignment import _last_masks, _last_frames

    panels = []
    for i in range(4):
        mask = _last_masks[i]
        frame = _last_frames[i]
        if mask is not None and frame is not None:
            overlay = frame.copy()
            green = np.zeros_like(overlay); green[:] = (0, 255, 0)
            mask_3ch = cv2.merge([mask, mask, mask])
            overlay = np.where(mask_3ch > 0,
                               cv2.addWeighted(overlay, 0.5, green, 0.5, 0),
                               overlay)
            cv2.putText(overlay, f'Cam {i+1}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            panels.append(overlay)
        elif mask is not None:
            panel = cv2.merge([mask, mask, mask])
            cv2.putText(panel, f'Cam {i+1}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            panels.append(panel)
        else:
            panels.append(None)

    if all(p is None for p in panels):
        print('No masks available yet - press G first to run a reconstruction.')
        return

    # Use the shape of the first available panel as reference
    ref = next(p for p in panels if p is not None)
    h, w = ref.shape[:2]
    for i in range(4):
        if panels[i] is None:
            blank = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(blank, f'Cam {i+1} (no data)', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
            panels[i] = blank

    top = np.hstack([panels[0], panels[1]])
    bot = np.hstack([panels[2], panels[3]])
    mosaic = np.vstack([top, bot])

    cv2.imshow('All Camera Masks', mosaic)
    cv2.waitKey(1)


def _start_sequence(mode='union'):
    """Start the sequential demo: cam1 → cam2 → cam3 → cam4.

    mode='union'     (H) – additive, each step adds another camera's voxels.
    mode='intersect'  (J) – iterative carving, each step intersects one more camera.
    """
    global _seq_active, _seq_cam_index, _seq_next_time, _seq_mode
    _seq_active = True
    _seq_cam_index = 0
    _seq_next_time = 0.0
    _seq_mode = mode
    label = 'ADDITIVE (union)' if mode == 'union' else 'ITERATIVE (intersect)'
    print(f'=== Sequential camera demo started – {label} – {_SEQ_DELAY:.0f} s per step ===')


def _seq_advance():
    """Show voxels for cameras 0..seq_cam_index, then schedule the next."""
    global cube, _seq_active, _seq_cam_index, _seq_next_time, _last_voxel_positions

    ci = _seq_cam_index
    cams_so_far = list(range(ci + 1))
    label = 'union' if _seq_mode == 'union' else 'intersect'
    print(f'[SEQ-{label.upper()}] Cameras {[c+1 for c in cams_so_far]}  '
          f'(step {ci + 1}/4)  –  next in {_SEQ_DELAY:.0f} s')

    positions, colors = set_voxel_positions(
        config['world_width'], config['world_height'], config['world_width'],
        debug_cams=cams_so_far, debug_mode=_seq_mode)
    cube.set_multiple_positions(positions, colors)
    _last_voxel_positions = positions
    _snap_camera_to(ci)

    _seq_cam_index += 1
    if _seq_cam_index >= 4:
        _seq_next_time = glfw.get_time() + _SEQ_DELAY
        _seq_active = False
        print(f'[SEQ-{label.upper()}] Sequence complete.')
    else:
        _seq_next_time = glfw.get_time() + _SEQ_DELAY


def _stop_sequence():
    global _seq_active
    _seq_active = False


def _playback_tick():
    """Advance to the next video frame and reconstruct voxels."""
    global cube, _playback_next_time, _last_voxel_positions
    positions, colors = set_voxel_positions(
        config['world_width'], config['world_height'], config['world_width'],
        debug_cam=_debug_cam_index)
    cube.set_multiple_positions(positions, colors)
    _last_voxel_positions = positions
    if _debug_cam_index >= 0:
        _snap_camera_to(_debug_cam_index)
    _playback_next_time = glfw.get_time() + 1.0 / _PLAYBACK_FPS


def _toggle_orbit():
    """Toggle orbit camera mode on/off."""
    global _orbit_active, _orbit_angle, _orbit_center, _orbit_radius
    global _orbit_pitch, _orbit_pitch_target, _orbit_speed, _orbit_speed_target
    _orbit_active = not _orbit_active
    if _orbit_active:
        # Compute orbit centre from last generated voxel positions
        if _last_voxel_positions and len(_last_voxel_positions) > 0:
            pts = _last_voxel_positions
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            zs = [p[2] for p in pts]
            _orbit_center = glm.vec3(
                sum(xs) / len(xs),
                sum(ys) / len(ys),
                sum(zs) / len(zs))
        else:
            _orbit_center = glm.vec3(0, 0, 0)

        # Derive initial angle from current camera position
        dx = camera.position.x - _orbit_center.x
        dz = camera.position.z - _orbit_center.z
        _orbit_angle = math.atan2(dz, dx)
        _orbit_radius = max(30.0, glm.length(camera.position - _orbit_center))
        _orbit_pitch = 30.0
        _orbit_pitch_target = 30.0
        _orbit_speed = 0.5
        _orbit_speed_target = 0.5
        print(f'\U0001f504  Orbit mode ON  (centre {_orbit_center.x:.0f}, '
              f'{_orbit_center.y:.0f}, {_orbit_center.z:.0f}  r={_orbit_radius:.0f})')
    else:
        print('\U0001f504  Orbit mode OFF')


def _orbit_update(dt):
    """Advance the orbit camera each frame (called from main loop)."""
    global _orbit_angle, _orbit_speed, _orbit_pitch
    if not _orbit_active:
        return

    # Lerp speed and pitch towards their targets
    t = min(1.0, _ORBIT_LERP_RATE * dt)
    _orbit_speed += (_orbit_speed_target - _orbit_speed) * t
    _orbit_pitch += (_orbit_pitch_target - _orbit_pitch) * t

    _orbit_angle += _orbit_speed * dt

    pitch_rad = math.radians(_orbit_pitch)
    cam_x = _orbit_center.x + _orbit_radius * math.cos(pitch_rad) * math.cos(_orbit_angle)
    cam_y = _orbit_center.y + _orbit_radius * math.sin(pitch_rad)
    cam_z = _orbit_center.z + _orbit_radius * math.cos(pitch_rad) * math.sin(_orbit_angle)

    camera.position = glm.vec3(cam_x, cam_y, cam_z)

    # Point camera at orbit centre
    to_center = glm.normalize(_orbit_center - camera.position)
    camera.yaw = math.degrees(math.atan2(to_center.z, to_center.x))
    camera.pitch = math.degrees(math.asin(max(-1.0, min(1.0, to_center.y))))
    camera.update_vectors()


def _toggle_playback():
    """Toggle spacebar playback on/off."""
    global _playback_active, _playback_next_time
    if not _g_pressed_once:
        print('Press G first to generate initial voxels before playback.')
        return
    _playback_active = not _playback_active
    if _playback_active:
        _playback_next_time = 0.0
        print(f'▶  Playback started ({_PLAYBACK_FPS:.0f} fps target)')
    else:
        print('⏸  Playback paused')


def key_callback(window, key, scancode, action, mods):
    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        _close_debug_overlay()
        glfw.set_window_should_close(window, glfw.TRUE)
    if key == glfw.KEY_G and action == glfw.PRESS:
        global cube, _g_pressed_once, _last_voxel_positions
        _g_pressed_once = True
        positions, colors = set_voxel_positions(
            config['world_width'], config['world_height'], config['world_width'],
            debug_cam=_debug_cam_index)
        cube.set_multiple_positions(positions, colors)
        _last_voxel_positions = positions
        if _debug_cam_index >= 0:
            _snap_camera_to(_debug_cam_index)

    if key == glfw.KEY_SPACE and action == glfw.PRESS:
        _toggle_playback()

    for i, k in enumerate([glfw.KEY_1, glfw.KEY_2, glfw.KEY_3, glfw.KEY_4]):
        if key == k and action == glfw.PRESS:
            _snap_camera_to(i)
            return

    if key in (glfw.KEY_0, glfw.KEY_5) and action == glfw.PRESS:
        _close_debug_overlay()
        _stop_sequence()
        print('Debug overlay closed – free camera mode.')

    if key == glfw.KEY_O and action == glfw.PRESS:
        _toggle_orbit()

    if _orbit_active:
        global _orbit_speed_target, _orbit_pitch_target
        if key == glfw.KEY_UP and action == glfw.PRESS:
            _orbit_speed_target += _ORBIT_SPEED_STEP
            print(f'  Orbit speed target → {_orbit_speed_target:.2f} rad/s')
        if key == glfw.KEY_DOWN and action == glfw.PRESS:
            _orbit_speed_target = max(0.05, _orbit_speed_target - _ORBIT_SPEED_STEP)
            print(f'  Orbit speed target → {_orbit_speed_target:.2f} rad/s')
        if key == glfw.KEY_RIGHT and action == glfw.PRESS:
            _orbit_pitch_target = min(85.0, _orbit_pitch_target + _ORBIT_PITCH_STEP)
            print(f'  Orbit pitch target → {_orbit_pitch_target:.1f}°')
        if key == glfw.KEY_LEFT and action == glfw.PRESS:
            _orbit_pitch_target = max(-85.0, _orbit_pitch_target - _ORBIT_PITCH_STEP)
            print(f'  Orbit pitch target → {_orbit_pitch_target:.1f}°')

    if key == glfw.KEY_H and action == glfw.PRESS:
        _start_sequence(mode='union')

    if key == glfw.KEY_J and action == glfw.PRESS:
        _start_sequence(mode='intersect')

    if key == glfw.KEY_M and action == glfw.PRESS:
        _show_all_masks()


def mouse_move(win, pos_x, pos_y):
    global firstTime, camera, lastPosX, lastPosY
    if firstTime:
        lastPosX = pos_x
        lastPosY = pos_y
        firstTime = False

    if not _orbit_active:
        camera.rotate(pos_x - lastPosX, lastPosY - pos_y)
    lastPosX = pos_x
    lastPosY = pos_y


def move_input(win, time):
    if _orbit_active:
        return
    if glfw.get_key(win, glfw.KEY_W) == glfw.PRESS:
        camera.move_top(time)
    if glfw.get_key(win, glfw.KEY_S) == glfw.PRESS:
        camera.move_bottom(time)
    if glfw.get_key(win, glfw.KEY_A) == glfw.PRESS:
        camera.move_left(time)
    if glfw.get_key(win, glfw.KEY_D) == glfw.PRESS:
        camera.move_right(time)


if __name__ == '__main__':
    main()
