import cv2
import mediapipe as mp
import numpy as np
from math import sqrt
import os
import random
import imageio
import json
# Initialize MediaPipe hands detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3, min_tracking_confidence=0.5)

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        ) 



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

def detect_hand_bbox_and_keypoints_hands(frame):
    """Use MediaPipe to detect hand bounding boxes and keypoints in a frame."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    hand_data = []  # To store information for each hand detected

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness_info in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Get hand detection confidence score
            detection_score = handedness_info.classification[0].score
            print('detection_score:', detection_score)

            # Get hand bounding box
            bbox = calculate_hand_bbox(hand_landmarks, frame.shape)
            # Get landmark confidence scores
            hand_data.append((bbox, hand_landmarks, detection_score))
            
    return hand_data

def detect_hand_bbox_and_keypoints_holistic(frame, visibility_threshold = 0.1):
    """Use MediaPipe to detect hand bounding boxes and keypoints in a frame."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(frame_rgb)

    hand_data = []  # To store information for each hand detected
    
    if results.left_hand_landmarks:

        # Check visibility for each landmark in the left hand
        left_hand_valid = all(landmark.visibility >= visibility_threshold for landmark in results.left_hand_landmarks.landmark)
        # for landmark in results.left_hand_landmarks.landmark:
        #     print('left landmark.visibility', landmark.visibility)
        if left_hand_valid:
            bbox = calculate_hand_bbox(results.left_hand_landmarks, frame.shape)
            # Collect visibility values for each landmark
            visibilities = [landmark.visibility for landmark in results.left_hand_landmarks.landmark]
            avg_score = sum(visibilities) / len(visibilities)
            hand_data.append((bbox, results.left_hand_landmarks, avg_score))
        # else:
        #     print('left hand not valid')
    
    if results.right_hand_landmarks:
        # for landmark in results.right_hand_landmarks.landmark:
        #     print('right landmark.visibility', landmark.visibility)
        # Check visibility for each landmark in the right hand
        right_hand_valid = all(landmark.visibility >= visibility_threshold for landmark in results.right_hand_landmarks.landmark)
        if right_hand_valid:
            bbox = calculate_hand_bbox(results.right_hand_landmarks, frame.shape)
            # Collect visibility values for each landmark
            visibilities = [landmark.visibility for landmark in results.right_hand_landmarks.landmark]
            avg_score = sum(visibilities) / len(visibilities)
            hand_data.append((bbox, results.right_hand_landmarks, avg_score))

            
    return hand_data

def parse_openpose_hand_bbox_and_keypoints(frame, json_filepath, confidence_threshold = 0.3):
    """Parse OpenPose JSON output to detect hand bounding boxes and keypoints."""
    hand_data = []  # To store information for each hand detected
    with open(json_filepath, 'r') as f:
        try:            
            data = json.load(f)
            left_hand_data = data['people'][0]['hand_left_keypoints_2d']
            right_hand_data = data['people'][0]['hand_right_keypoints_2d']

            # Extract left hand keypoints
            left_hand_keypoints = []
            for i in range(0, len(left_hand_data), 3):
                x, y, score = left_hand_data[i:i+3]
                if score < 0.1:
                    continue
                left_hand_keypoints.append((x, y, score))
            
            # Extract right hand keypoints
            right_hand_keypoints = []
            for i in range(0, len(right_hand_data), 3):
                x, y, score = right_hand_data[i:i+3]
                if score < 0.1:
                    continue
                right_hand_keypoints.append((x, y, score))
        except:
            print('Error parsing JSON file:', json_filepath)
            return hand_data
        
    if len(left_hand_keypoints):
        avg_left_hand_score = sum([kp[2] for kp in left_hand_keypoints]) / len(left_hand_keypoints)
    else:
        avg_left_hand_score = 0

    if len(right_hand_keypoints):
        avg_right_hand_score = sum([kp[2] for kp in right_hand_keypoints]) / len(right_hand_keypoints)
    else:
        avg_right_hand_score = 0

    if avg_left_hand_score >= confidence_threshold:
        left_hand_bbox = calculate_hand_bbox_openpose(left_hand_keypoints, frame.shape)
        hand_data.append((left_hand_bbox, left_hand_keypoints, avg_left_hand_score))
    
    if avg_right_hand_score >= confidence_threshold:
        right_hand_bbox = calculate_hand_bbox_openpose(right_hand_keypoints, frame.shape)
        hand_data.append((right_hand_bbox, right_hand_keypoints, avg_right_hand_score))

    return hand_data

def calculate_hand_bbox_openpose(keypoints, frame_shape):
    """Calculate the bounding box of the hand based on keypoints."""
    h, w, _ = frame_shape
    x_coords = [kp[0] for kp in keypoints]
    y_coords = [kp[1] for kp in keypoints]
    xmin, xmax = min(x_coords), max(x_coords)
    ymin, ymax = min(y_coords), max(y_coords)
    
    # bbox_center = ((xmin + xmax) / 2, (ymin + ymax) / 2)
    # bbox_size = (xmax - xmin, ymax - ymin)
    bbox = (xmin, ymin, xmax, ymax)
    return bbox


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

def select_key_frames(video_path, num_key_frames=4, sharpness_threshold=150.0, confidence_threshold=0.5, dector='mediapipe_hands'):
    frames = extract_frames(video_path)
    detected_frames = []
    movements = []
    visibility_threshold = 0.4
    valid_keypoints_ratio = 0.7

    # Step 1: Filter frames with detected hands, calculate bbox movements, and sharpness
    for i, frame in enumerate(frames):

        
        if dector == 'openpose':
            video_basename = os.path.basename(video_path).split('.')[0]
            json_filename = f'{video_basename}_{i:012d}_keypoints.json'
            json_filepath = os.path.join('openpose_json', json_filename)
            if not os.path.exists(json_filepath):
                return []
            hand_data = parse_openpose_hand_bbox_and_keypoints(frame, json_filepath, confidence_threshold=confidence_threshold)
        elif dector == 'holistic':
            hand_data = detect_hand_bbox_and_keypoints_holistic(frame, visibility_threshold=visibility_threshold)
        else:
            hand_data = detect_hand_bbox_and_keypoints_hands(frame)
        # print('hand_data:', hand_data)
        if len(hand_data):
            all_hands_valid = True  # Flag to track if all detected hands meet the requirements
            
            # Process each detected hand            
            for bbox, hand_landmarks, detection_score in hand_data:
                # Check if the hand detection score is above a threshold (e.g., 0.5)
                if detection_score < confidence_threshold:
                    all_hands_valid = False
                    break  # If any hand doesn't meet the detection score threshold, skip this frame


                # Extract the hand region from the frame
                bbox = [int(coord) for coord in bbox]  # Convert to int
                xmin, ymin, xmax, ymax = bbox
                hand_region = frame[ymin:ymax, xmin:xmax]
                
                # Calculate sharpness of the hand region
                sharpness = calculate_sharpness(hand_region)
                # print(f'{video_path}, sharpness:', sharpness)
                # Check if the hand's sharpness is above the threshold
                if sharpness <= sharpness_threshold:
                    all_hands_valid = False
                    break  # If any hand doesn't meet the sharpness threshold, skip this frame
            
            # If all detected hands meet the conditions, add this frame to the detected_frames
            if all_hands_valid:
                # Assuming we take the center of the first hand's bbox as a reference (optional)
                bbox_center = ((xmin + xmax) / 2, (ymin + ymax) / 2)
                detected_frames.append((i, frame, bbox_center))

    if len(detected_frames) == 0:
        print('No frames with valid hand detections found.')
        return []

    # Calculate average movement between consecutive detected frames
    for i in range(1, len(detected_frames)):
        prev_bbox = detected_frames[i-1][2]  # Previous bounding box center
        curr_bbox = detected_frames[i][2]  # Current bounding box center
        movement = calculate_movement(prev_bbox, curr_bbox)
        movements.append(movement)
    avg_movement = np.mean(movements) if movements else 0

    # print('detected_frames:', detected_frames)
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
    confidence_threshold = 0.6
    sharpness_threshold = 150.0
    dector = 'mediapipe_hands'

    out_dir = f'output_{dector}'
    os.makedirs(out_dir, exist_ok=True)
    #random select 10 videos
    # random.seed(0)
    # random.shuffle(video_names)
    # video_names = video_names[:10]

    for i, video_name in enumerate(video_names):
        print(f'Processing video: {video_name}, {i}/{len(video_names)}\n')
        video_path = os.path.join(video_dir, video_name)
        # new_video_path = os.path.join(test_video_dir, video_name)
        # os.system(f'cp {video_path} {new_video_path}')
        key_frames = select_key_frames(video_path, num_key_frames=4, confidence_threshold = confidence_threshold, dector = dector)
        if len(key_frames) == 0:            
            continue
        video_basename = os.path.basename(video_path).split('.')[0]

        frames = np.concatenate(key_frames, axis=1)
        cv2.imwrite(f'{out_dir}/{video_basename}.jpg', frames)
        # break   
        
