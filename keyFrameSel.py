import cv2
import mediapipe as mp
import numpy as np
from math import sqrt
import os
import random

# Initialize MediaPipe hands detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

def calculate_movement(bbox1, bbox2):
    """Calculate the movement between two bounding box centers."""
    x1, y1 = bbox1
    x2, y2 = bbox2
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def extract_frames(video_path):
    """Extract frames from the video and return them as a list."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def detect_hand_bbox_and_keypoints(frame):
    """Use MediaPipe to detect hand bounding boxes and keypoints in a frame."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        bbox = calculate_hand_bbox(hand_landmarks, frame.shape)
        return bbox, hand_landmarks
    return None, None

def calculate_hand_bbox(landmarks, frame_shape):
    """Calculate the bounding box of the hand based on keypoints."""
    h, w, _ = frame_shape
    x_coords = [landmark.x * w for landmark in landmarks.landmark]
    y_coords = [landmark.y * h for landmark in landmarks.landmark]
    xmin, xmax = min(x_coords), max(x_coords)
    ymin, ymax = min(y_coords), max(y_coords)
    
    # bbox_center = ((xmin + xmax) / 2, (ymin + ymax) / 2)
    # bbox_size = (xmax - xmin, ymax - ymin)
    bbox = (xmin, ymin, xmax, ymax)
    return bbox

def calculate_sharpness(frame):
    """Calculate the sharpness of an image using the Laplacian method."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var

def select_key_frames(video_path, num_key_frames=4, sharpness_threshold=80.0):
    frames = extract_frames(video_path)
    detected_frames = []
    movements = []
    
    # Step 1: Filter frames with detected hands and calculate bbox movements and sharpness
    for i, frame in enumerate(frames):
        bbox, hand_landmarks = detect_hand_bbox_and_keypoints(frame)
        if bbox is not None and hand_landmarks is not None:
            # Extract the hand region from the frame
            bbox = [int(coord) for coord in bbox]  # Convert to int
            xmin, ymin, xmax, ymax = bbox
            hand_region = frame[ymin:ymax, xmin:xmax]
            bbox_center = ((xmin + xmax) / 2, (ymin + ymax) / 2)
            # Calculate sharpness of the hand region
            sharpness = calculate_sharpness(hand_region)

            # Check if the frame's sharpness is above the threshold
            if sharpness > sharpness_threshold:
                detected_frames.append((i, frame, bbox_center))

    # Calculate average movement between consecutive detected frames
    for i in range(1, len(detected_frames)):
        prev_bbox = detected_frames[i-1][2]  # Previous bounding box center
        curr_bbox = detected_frames[i][2]  # Current bounding box center
        movement = calculate_movement(prev_bbox, curr_bbox)
        movements.append(movement)
    avg_movement = np.mean(movements) if movements else 0

    # Step 2: Select key frames based on movement compared to average movement
    key_frame_indices = [detected_frames[0][0]]  # Start with the first frame as baseline
    baseline_bbox = detected_frames[0][2]  # Initial baseline bbox center

    for i in range(1, len(detected_frames)):
        curr_bbox = detected_frames[i][2]
        movement = calculate_movement(baseline_bbox, curr_bbox)
        
        # If movement exceeds threshold, select this frame as a key frame and update baseline
        if movement > avg_movement:
            key_frame_indices.append(detected_frames[i][0])
            baseline_bbox = curr_bbox  # Update baseline

    # Step 3: Uniformly sample 4-5 frames from the selected key frames
    num_key_frames = min(num_key_frames, len(key_frame_indices))  # Adjust if fewer key frames available
    sampled_key_frames_indices = np.linspace(0, len(key_frame_indices) - 1, num_key_frames, dtype=int)
    sampled_key_frames = [frames[key_frame_indices[i]] for i in sampled_key_frames_indices]
    
    return sampled_key_frames


if __name__ == '__main__':

    # Usage Example
    # video_dir = '/scratch/rhong5/dataset/signLanguage/WLASL/raw_videos'
    video_dir = 'test_video'
    video_names = os.listdir(video_dir)
    # test_video_dir = 'test_video'
    # os.makedirs(test_video_dir, exist_ok=True)

    out_dir = 'output'
    os.makedirs(out_dir, exist_ok=True)
    #random select 10 videos
    random.seed(0)
    random.shuffle(video_names)
    video_names = video_names[:10]

    for i, video_name in enumerate(video_names):
        print(f'Processing video: {video_name}, {i}/{len(video_names)}')
        video_path = os.path.join(video_dir, video_name)
        # new_video_path = os.path.join(test_video_dir, video_name)
        # os.system(f'cp {video_path} {new_video_path}')
        key_frames = select_key_frames(video_path, num_key_frames=4)
        video_basename = os.path.basename(video_path).split('.')[0]

        frames = np.concatenate(key_frames, axis=1)
        cv2.imwrite(f'{out_dir}/{video_basename}.jpg', frames)
