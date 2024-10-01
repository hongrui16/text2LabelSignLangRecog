# text2LabelSignLangRecog

## Key Frame Extraction(Videos and Corresponding Images)
The key frames are extracted based on the hand's bounding box, key points, confidence of bounding boxes or key points, motion, and image blurriness.

---
---

## Part 1 (The key points are detected by MediaPipe in this part.)

### Video 1
**Video (click the image below to download the video):**  
[![04430.mp4](output/mediapipe_hands/04430.jpg)](test_video/04430.mp4)

**Extracted Image:**  
![04430.jpg](output/mediapipe_hands/04430.jpg)

---

### Video 2
**Video (click the image below to download the video):**  
[![06650.mp4](output/mediapipe_hands/06650.jpg)](test_video/06650.mp4)

**Extracted Image:**  
![06650.jpg](output/mediapipe_hands/06650.jpg)

---

### Video 3
**Video (click the image below to download the video):**  
[![07475.mp4](output/mediapipe_hands/07475.jpg)](test_video/07475.mp4)

**Extracted Image:**  
![07475.jpg](output/mediapipe_hands/07475.jpg)

---

### Video 4
**Video (click the image below to download the video):**  
[![15606.mp4](output/mediapipe_hands/15606.jpg)](test_video/15606.mp4)

**Extracted Image:**  
![15606.jpg](output/mediapipe_hands/15606.jpg)

---

### Video 5
**Video (click the image below to download the video):**  
[![30385.mp4](output/mediapipe_hands/30385.jpg)](test_video/30385.mp4)

**Extracted Image:**  
![30385.jpg](output/mediapipe_hands/30385.jpg)

---

### Video 6
**Video (click the image below to download the video):**  
[![44900.mp4](output/mediapipe_hands/44900.jpg)](test_video/44900.mp4)

**Extracted Image:**  
![44900.jpg](output/mediapipe_hands/44900.jpg)

---

### Video 7
**Video (click the image below to download the video):**  
[![50508.mp4](output/mediapipe_hands/50508.jpg)](test_video/50508.mp4)

**Extracted Image:**  
![50508.jpg](output/mediapipe_hands/50508.jpg)

---

### Video 8
**Video (click the image below to download the video):**  
[![53706.mp4](output/mediapipe_hands/53706.jpg)](test_video/53706.mp4)

**Extracted Image:**  
![53706.jpg](output/mediapipe_hands/53706.jpg)

---

### Video 9
**Video (click the image below to download the video):**  
[![65140.mp4](output/mediapipe_hands/65140.jpg)](test_video/65140.mp4)

**Extracted Image:**  
![65140.jpg](output/mediapipe_hands/65140.jpg)

---

### Video 10
**Video (click the image below to download the video):**  
[![65388.mp4](output/mediapipe_hands/65388.jpg)](test_video/65388.mp4)

**Extracted Image:**  
![65388.jpg](output/mediapipe_hands/65388.jpg)

---
---

## Part 2 (The key points are detected by OpenPose in this part.)

### Video 2
**Video (click the image below to download the video):**  
[![06650.mp4](output/openpose/06650.jpg)](test_video/06650.mp4)

**Extracted Image:**  
![06650.jpg](output/openpose/06650.jpg)

---

### Video 4
**Video (click the image below to download the video):**  
[![15606.mp4](output/openpose/15606.jpg)](test_video/15606.mp4)

**Extracted Image:**  
![15606.jpg](output/openpose/15606.jpg)

---

### Video 5
**Video (click the image below to download the video):**  
[![30385.mp4](output/openpose/30385.jpg)](test_video/30385.mp4)

**Extracted Image:**  
![30385.jpg](output/openpose/30385.jpg)

---

### Video 9
**Video (click the image below to download the video):**  
[![65140.mp4](output/openpose/65140.jpg)](test_video/65140.mp4)

**Extracted Image:**  
![65140.jpg](output/openpose/65140.jpg)

---

### Video 10
**Video (click the image below to download the video):**  
[![65388.mp4](output/openpose/65388.jpg)](test_video/65388.mp4)

**Extracted Image:**  
![65388.jpg](output/openpose/65388.jpg)


---
---
## Summary

If we increase the threshold for key points confidence or the threshold for sharpness, both will filter out blury frames, resulting in a lack of dynamic motion.