import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import requests

sos = 0

# Function to send HTTP requests
def send_request():
    global sos
    while True:
        try:
            # Example payload (customize as needed)
            payload = {'sos': sos}
            print('sent : ' + str(sos))
            response = requests.post('http://localhost:5000/upd', json=payload)
            if sos: 
                sos = 0
            print(f"Request sent, status code: {response.status_code}")
        except Exception as e:
            print(f"An error occurred: {e}")
        
        # Wait for a specified interval before sending the next request
        time.sleep(2)  # Adjust the interval as needed

# Start sending requests in a separate thread
request_thread = threading.Thread(target=send_request, daemon=True)
request_thread.start()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def is_ok_sign(hand_landmarks):
    """ Determine if the detected hand landmarks correspond to the 'OK' sign. """
    # Access the landmarks
    if not hand_landmarks:
        return False
    
    # Extract the landmark positions
    landmarks = hand_landmarks.landmark

    # LANDMARKS
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]

    # Calculate distances
    thumb_index_dist = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
    thumb_middle_dist = np.sqrt((thumb_tip.x - middle_tip.x)**2 + (thumb_tip.y - middle_tip.y)**2)
    thumb_ring_dist = np.sqrt((thumb_tip.x - ring_tip.x)**2 + (thumb_tip.y - ring_tip.y)**2)
    thumb_pinky_dist = np.sqrt((thumb_tip.x - pinky_tip.x)**2 + (thumb_tip.y - pinky_tip.y)**2)

    # Heuristic values (these thresholds might need adjustments based on testing)
    # used heuristic --> The thumb tip should be close to the index finger tip and the other fingers should be curled -- Used Heuristic
    distance_threshold = 0.05

    if (thumb_index_dist < distance_threshold and
        thumb_middle_dist > 0.1 and
        thumb_ring_dist > 0.1 and
        thumb_pinky_dist > 0.1):
        return True
    return False

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            sent = False
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if is_ok_sign(hand_landmarks):
                    cv2.putText(frame, 'OK Sign Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    sos = 1
                    
        cv2.imshow('Hand Gesture Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
