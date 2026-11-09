"""
Live webcam filter: puts glasses on every face using OpenCV's YuNet face detector.
Also applies a subtle eye-enlargement effect (~10%) before drawing the glasses.
Needs in the same folder as this script:
  - glasses.png
  - the YuNet model file (downloaded automatically the first time)
Press 'q' to quit.
"""
import math
import os
import sys
import urllib.request
import cv2
import numpy as np

import kiosk  # touchscreen Back button, fullscreen and auto-back (see kiosk.py)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'face_detection_yunet_2023mar.onnx')
MODEL_URL = ('https://github.com/opencv/opencv_zoo/raw/main/models/'
             'face_detection_yunet/face_detection_yunet_2023mar.onnx')
GLASSES_PATH = os.path.join(SCRIPT_DIR, 'glasses.png')

WIDTH_SCALE = 2.2
LENS_Y_RATIO = 0.5
EYE_SCALE = 2  # 100% larger eyes

# Acceptance criteria settings
MIN_FACE_WIDTH = 80
MAX_FACE_WIDTH = 420
FRAME_CENTER_TOLERANCE = 0.45

def load_model() -> None:
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 100_000:
        return
    print('Downloading YuNet face model (one time only)...')
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print('Done.')

'''
def enlarge_eye_region(frame, eye_center, radius, scale=1.10):
    x, y = int(eye_center[0]), int(eye_center[1])
    r = int(radius)

    h, w = frame.shape[:2]
    x1, x2 = max(0, x - r), min(w, x + r)
    y1, y2 = max(0, y - r), min(h, y + r)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return frame

    rh, rw = roi.shape[:2]
    map_x = np.zeros((rh, rw), np.float32)
    map_y = np.zeros((rh, rw), np.float32)

    cx, cy = rw / 2.0, rh / 2.0
    max_r = min(cx, cy)

    for iy in range(rh):
        for ix in range(rw):
            dx = ix - cx
            dy = iy - cy
            dist = np.sqrt(dx * dx + dy * dy)

            if dist < max_r:
                factor = 1.0 - (dist / max_r)
                local_scale = 1.0 + (scale - 1.0) * factor
                map_x[iy, ix] = cx + dx / local_scale
                map_y[iy, ix] = cy + dy / local_scale
            else:
                map_x[iy, ix] = ix
                map_y[iy, ix] = iy

    frame[y1:y2, x1:x2] = cv2.remap(
        roi, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )
    return frame
'''
def enlarge_eye_region(frame, eye_center, radius, scale=1.40):
    x, y = int(eye_center[0]), int(eye_center[1])

    # Bigger source area
    src_r = int(radius * 2)

    h, w = frame.shape[:2]

    x1 = max(0, x - src_r)
    y1 = max(0, y - src_r)
    x2 = min(w, x + src_r)
    y2 = min(h, y + src_r)

    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return frame

    # Enlarge eye region
    enlarged = cv2.resize(
        roi,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )

    eh, ew = enlarged.shape[:2]

    # Destination region
    dx1 = int(x - ew / 2)
    dy1 = int(y - eh / 2)
    dx2 = dx1 + ew
    dy2 = dy1 + eh

    if dx1 < 0 or dy1 < 0 or dx2 > w or dy2 > h:
        return frame

    target = frame[dy1:dy2, dx1:dx2]

    # Soft circular mask
    mask = np.zeros((eh, ew), dtype=np.float32)

    cv2.ellipse(
        mask,
        (ew // 2, eh // 2),
        (int(ew * 0.40), int(eh * 0.35)),
        0,
        0,
        360,
        1.0,
        -1,
    )

    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    mask = mask[..., np.newaxis]

    blended = (
        enlarged.astype(np.float32) * mask +
        target.astype(np.float32) * (1.0 - mask)
    )

    frame[dy1:dy2, dx1:dx2] = blended.astype(np.uint8)

    return frame

def overlay_glasses(frame, glasses, left_eye, right_eye):
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    eye_dist = math.hypot(dx, dy)
    angle_deg = math.degrees(math.atan2(dy, dx))

    mid_x = (left_eye[0] + right_eye[0]) / 2
    mid_y = (left_eye[1] + right_eye[1]) / 2

    g_h, g_w = glasses.shape[:2]
    lens_center = (g_w / 2, g_h * LENS_Y_RATIO)

    scale = (eye_dist * WIDTH_SCALE) / g_w
    M = cv2.getRotationMatrix2D(lens_center, -angle_deg, scale)

    M[0, 2] += mid_x - lens_center[0]
    M[1, 2] += mid_y - lens_center[1]

    f_h, f_w = frame.shape[:2]
    warped = cv2.warpAffine(
        glasses, M, (f_w, f_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    alpha = warped[:, :, 3:4].astype(np.float32) / 255.0
    blended = frame * (1 - alpha) + warped[:, :, :3] * alpha
    return blended.astype(np.uint8)


def main():
    
    load_model()
    '''
    glasses = cv2.imread(GLASSES_PATH, cv2.IMREAD_UNCHANGED)
    if glasses is None:
        print(f'Error: Could not load {GLASSES_PATH}')
        return

    if glasses.shape[2] != 4:
        print('Error: glasses.png needs a transparent background (4 channels).')
        return
    '''
    backend = cv2.CAP_DSHOW if sys.platform.startswith('win') else cv2.CAP_ANY
    
    cap = cv2.VideoCapture(0, backend)

    if not cap.isOpened():
        print('Error: Could not open webcam.')
        return
    '''
    for i in range(10):
        cap = cv2.VideoCapture(i)
        
        if cap.isOpened():
            print(f"Camera {i}: Available")
            ret, frame = cap.read()
            if ret:
                cv2.imshow(f"Camera {i}", frame)
                cv2.waitKey(1000)
                cv2.destroyAllWindows()
            cap.release()
            break
        else:
            print(f"Camera {i}: Not available")
            '''
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    detector = cv2.FaceDetectorYN.create(
        MODEL_PATH, '', (640, 480), score_threshold=0.8
    )
    '''
    print()
    print("Press Y to enable the filter.")
    print("Press any other key to exit.")

    if input("> ").strip().lower() != "y":
        return
    
    print("Running.")
    print("1 = Big Eyes")
    print("2 = Glasses")
    print("3 = Big Eyes + Glasses")
    print("4 = No Filter")
    print("q = Quit")

    current_filter = 1
    '''
    print("Running. Press 'q' to quit.")
    last_faces = None
    screen = kiosk.Kiosk('Glasses Filter')



    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        detector.setInputSize((w, h))

        _, faces = detector.detect(frame)
        face_seen = faces is not None  # for the auto-back timer (before the last_faces fallback)
        if faces is not None:
            last_faces = faces
        elif last_faces is not None:
            faces = last_faces
        if faces is not None:
            for face in faces:
                face_width = face[2]

                if not (MIN_FACE_WIDTH <= face_width <= MAX_FACE_WIDTH):
                    continue
                face_x = face[0]
                face_center_x = face_x + face_width / 2

                normalized_offset = abs(
                    face_center_x - (w / 2)
                ) / (w / 2)

                if normalized_offset > FRAME_CENTER_TOLERANCE:
                    continue
                eye_a = (face[4], face[5])
                eye_b = (face[6], face[7])
                left_eye, right_eye = sorted([eye_a, eye_b])

                eye_radius = max(12, int(math.hypot(
                    right_eye[0] - left_eye[0],
                    right_eye[1] - left_eye[1]
                ) * 0.18))

                #frame = enlarge_eye_region(frame, left_eye, eye_radius, EYE_SCALE)
                #frame = enlarge_eye_region(frame, right_eye, eye_radius, EYE_SCALE)
                # Filter 1 - Big Eyes
                #if current_filter == 1:

                frame = enlarge_eye_region(
                    frame,
                    left_eye,
                    eye_radius,
                    EYE_SCALE
                )

                frame = enlarge_eye_region(
                    frame,
                    right_eye,
                    eye_radius,
                    EYE_SCALE
                )

                # Filter 2 - Glasses
                '''
                elif current_filter == 2:

                    frame = overlay_glasses(
                        frame,
                        glasses,
                        left_eye,
                        right_eye
                    )

                # Filter 3 - Big Eyes + Glasses
                elif current_filter == 3:

                    frame = enlarge_eye_region(
                        frame,
                        left_eye,
                        eye_radius,
                        EYE_SCALE
                    )

                    frame = enlarge_eye_region(
                        frame,
                        right_eye,
                        eye_radius,
                        EYE_SCALE
                    )

                    frame = overlay_glasses(
                        frame,
                        glasses,
                        left_eye,
                        right_eye
                    )

                # Filter 4 - No Filter
                elif current_filter == 4:
                    pass
                '''
            #   frame = overlay_glasses(frame, glasses, left_eye, right_eye)
        '''
        filter_names = {
            1: "Big Eyes",
            2: "Glasses",
            3: "Big Eyes + Glasses",
            4: "No Filter"
        }

        cv2.putText(
            frame,
            f"Filter: {filter_names[current_filter]}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        '''

        # Shows the frame (like cv2.imshow) and returns the key pressed or tapped
        key = screen.show(frame, person_seen=face_seen)
        '''
        key = cv2.waitKey(1) & 0xFF

        if key == ord('1'):
            current_filter = 1

        elif key == ord('2'):
            current_filter = 2

        elif key == ord('3'):
            current_filter = 3

        elif key == ord('4'):
            current_filter = 4

        elif key == ord('q'):
            break
        '''
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
