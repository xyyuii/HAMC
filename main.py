import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from collections import deque

# Initializing the video
video = cv2.VideoCapture(1)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()

# Initializing audio utils
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Volume range variables initialized
vol_range = volume.GetVolumeRange()
min_vol = vol_range[0]
max_vol = vol_range[1]

# Smoothing
hand_y_positions = deque(maxlen=10)
prev_hand_y = None
play_pause_triggered = False

# When the index and thumb are together, if the hand moves up, the volume moves up, and vice versa.
# When the ring and thumb are together, the video play/pauses.
while True:
    success, frame = video.read()
    if success:
        RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(RGB_frame)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get the landmarks for the index finger, thumb, and ring finger
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                
                # Calculate the distance between the index finger tip and thumb tip, and ring finter tip and thumb finger tip.
                distance_index_thumb = np.sqrt((index_finger_tip.x - thumb_tip.x) ** 2 + (index_finger_tip.y - thumb_tip.y) ** 2)
                distance_ring_thumb = np.sqrt((ring_finger_tip.x - thumb_tip.x) ** 2 + (ring_finger_tip.y - thumb_tip.y) ** 2)
                
                # If the distance between the index finger and thumb is below a certain amount, the if statement runs
                if distance_index_thumb < 0.05:
                    # Vertical position of the hand
                    hand_y = (index_finger_tip.y + thumb_tip.y) / 2
                    
                    if prev_hand_y is not None:
                        # Calculate the change in the vertical position
                        delta_y = prev_hand_y - hand_y
                        
                        # Current volume level
                        current_vol = volume.GetMasterVolumeLevel()
                        
                        # Adjust the volume based on the change in the vertical position
                        new_vol = np.clip(current_vol + delta_y * (max_vol - min_vol) * 0.8, min_vol, max_vol)
                        volume.SetMasterVolumeLevel(new_vol, None)
                    
                    # Update the previous hand_y position
                    prev_hand_y = hand_y
                else:
                    prev_hand_y = None  # Reset if fingers are not together
                
                # If the distance between the ring finger and thumb is below a certain amount, the if statement runs
                if distance_ring_thumb < 0.05:
                    if not play_pause_triggered:
                        # Play or pause the media activated
                        pyautogui.press('playpause')
                        play_pause_triggered = True
                else:
                    play_pause_triggered = False  # Reset if fingers are not together
                    
        cv2.imshow("HAMC v1.1 (Press Q to end)", frame)
        if cv2.waitKey(1) == ord('q'):
            break

video.release()
cv2.destroyAllWindows()
