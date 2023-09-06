import cv2
import torch
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import shutil
from scipy.interpolate import RectBivariateSpline


# shutil.rmtree(torch.hub.get_dir(), ignore_errors=True)


# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(static_image_mode=False)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False)
# #download the model
midas = torch.hub.load('intel-isl/MiDaS','MiDaS_small')
midas.to('cpu')
midas.eval()

#Process image
transforms = torch.hub.load('intel-isl/MiDaS','transforms')
transform = transforms.small_transform

alpha = 0.2
previous_depth = 0.0
depth_scale = 1.0

#Applying exponential moving average filter
def apply_ema_filter(current_depth):
    global previous_depth
    filtered_depth = alpha * current_depth + (1 - alpha) * previous_depth
    previous_depth = filtered_depth  # Update the previous depth value
    return filtered_depth


#Define depth to distance
def depth_to_distance(depth_value,depth_scale):
    return 1.0 / (depth_value*depth_scale)

def depth_to_distance1(depth_value,depth_scale):
    return -1.0 / (depth_value*depth_scale)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    results = hands.process(img)


    if results.multi_hand_landmarks is not None:
        # Draw Landmarks
        mp_drawing = mp.solutions.drawing_utils
        # mp_drawing.draw_landmarks(img, results.multi_hand_landmarks.landmark, mp_hands.HAND_CONNECTIONS)

        
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            x_coords = []
            y_coords = []
            z_coords = []

            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for landmark in hand_landmarks.landmark:
                x_coords.append(landmark.x)
                y_coords.append(landmark.y)
                z_coords.append(landmark.z)
                landmarks.append((landmark.x, landmark.y, landmark.z))

        mid_x = sum(x_coords) / len(x_coords)
        mid_y = sum(y_coords) / len(y_coords)
        mid_z = sum(z_coords) / len(z_coords)
        # mid_point=0
        # mid_point = ((waist_landmarks[0].x + waist_landmarks[1].x) / 2, (waist_landmarks[0].y + waist_landmarks[1].y) / 2,(waist_landmarks[0].z + waist_landmarks[1].z) /2)
        # print(mid_point)
        # _,mid_x,mid_y= 0,0,0


        imgbatch = transform(img).to('cpu')

        # Making a prediction
        with torch.no_grad():
            prediction = midas(imgbatch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode='bicubic',
                align_corners=False
            ).squeeze()

        output = prediction.cpu().numpy()
        output_norm = cv2.normalize(output, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        h, w = output_norm.shape
        x_grid = np.arange(w)
        y_grid = np.arange(h)

        spline = RectBivariateSpline(y_grid, x_grid, output_norm)
        depth_mid_filt = spline(mid_y,mid_x)
        depth_midas = depth_to_distance(depth_mid_filt, depth_scale)
        depth_mid_filt = (apply_ema_filter(depth_midas)/10)[0][0]

        cv2.putText(img, "Depth in unit: " + str(
            np.format_float_positional(depth_mid_filt , precision=3)) , (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 3)

        cv2.imshow('Walking',img)

    if cv2.waitKey(1) &0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()