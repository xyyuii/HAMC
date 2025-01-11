import cv2
import mediapipe as mp
import numpy as np
import pycaw.pycaw as pycaw
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from collections import deque

# Video capture
video = cv2.VideoCapture(1)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
FRAME_WIDTH = 800
FRAME_HEIGHT = 600

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

vol_range = volume.GetVolumeRange()
min_vol = vol_range[0]
max_vol = vol_range[1]

hand_y_positions = deque(maxlen=10)

while True:
    success, frame = video.read()
    if success:
        RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(RGB_frame)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Finger landmarks
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                
                # Distance between the tip of the thumb and index finger.
                distance = np.sqrt((index_finger_tip.x - thumb_tip.x) ** 2 + (index_finger_tip.y - thumb_tip.y) ** 2)
                
                # If loop occurs when the distance between the two fingers is less than 0.05
                if distance < 0.05:
                    hand_y = (index_finger_tip.y + thumb_tip.y) / 2
                    
                    hand_y_positions.append(hand_y)
                    
                    avg_hand_y = np.mean(hand_y_positions)
                    
                    vol = np.interp(avg_hand_y, [0, 1], [max_vol, min_vol])
                    volume.SetMasterVolumeLevel(vol, None)
                    
        cv2.imshow("HAMC Sample Window (Press Q to end)", frame)
        if cv2.waitKey(1) == ord('q'):
            break

video.release()
cv2.destroyAllWindows()
