import cv2
import mediapipe as mp

video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

FRAME_WIDTH = cv2.CAP_PROP_FRAME_WIDTH
FRAME_HEIGHT = cv2.CAP_PROP_FRAME_HEIGHT

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()
all_landmark = []

while True:
    success, frame = video.read()
    if success:
        RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(RGB_frame)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                #print(hand_landmarks)
                for ids, landmrk in enumerate(hand_landmarks.landmark):
                    cx, cy = landmrk.x * FRAME_WIDTH, landmrk.y*FRAME_HEIGHT
                    #print(ids, cx, cy)
                    all_landmark.append((id, cx, cy))
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imshow("test", frame)
        if cv2.waitKey(1) == ord('q'):
            break
video.release()
