import cv2
import torch
import time
import numpy as np
import mediapipe as mp

# Initialize Mediapipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)

# Initialize MiDaS for depth estimation
model_type = "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# Initialize video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Read a frame from the camera
    success, frame = cap.read()
    
    if not success:
        break
    
    start_time = time.time()
    
    # Convert the frame to RGB format for hand pose detection
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect hand poses using Mediapipe
    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
        hand_results = hands.process(frame_rgb)
        # hand_results = hand_context.process(frame_rgb)
    
    # Convert the frame to depth using MiDaS
    frame_depth = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(frame_depth)
    with torch.no_grad():
        depth_prediction = midas(input_batch)
        depth_prediction = torch.nn.functional.interpolate(depth_prediction.unsqueeze(1), size=frame.shape[:2],
                                                           mode="bicubic", align_corners=False,).squeeze()
        depth_map = depth_prediction.cpu().numpy()
        depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    
    # Overlay hand pose points on the depth image
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(depth_map, (x, y), 5, (0, 0, 255), -1)
    
    # Calculate and display FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    
    # Display the depth image with hand pose points
    depth_map = (depth_map * 255).astype(np.uint8)
    depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)
    cv2.imshow('Depth Image with Hand Pose', cv2.addWeighted(frame, 0.1, depth_map, 0.9, 0))
    
    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
