import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands and Drawing Utilities
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Initialize Webcam
cap = cv2.VideoCapture(0)

# Memory for storing gestures
gestures = {}  # Format: {"Gesture Name": [[x1, y1, z1], [x2, y2, z2], ...]}

# Function to calculate similarity between gestures
def is_matching_gesture(landmarks, saved_landmarks, threshold=20):
    # Compare the distance between corresponding points
    if len(landmarks) != len(saved_landmarks):
        return False
    total_distance = 0
    for lm, saved_lm in zip(landmarks, saved_landmarks):
        total_distance += np.linalg.norm(np.array(lm) - np.array(saved_lm))
    return total_distance / len(landmarks) < threshold

# Set up Hand Detection
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a mirror-like effect
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect hands
        results = hands.process(frame_rgb)

        # List to store the current hand landmarks
        current_landmarks = []

        # If hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks and connections on the hand
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get coordinates of landmarks
                for lm in hand_landmarks.landmark:
                    h, w, _ = frame.shape
                    current_landmarks.append([lm.x * w, lm.y * h, lm.z])

            # Check for matching gestures
            matched_gesture = None
            for gesture_name, saved_landmarks in gestures.items():
                if is_matching_gesture(current_landmarks, saved_landmarks):
                    matched_gesture = gesture_name
                    break

            # Display the matched gesture
            if matched_gesture:
                cv2.putText(frame, f"Gesture: {matched_gesture}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Instructions for memorizing gestures
        cv2.putText(frame, "Press 's' to save gesture, 'q' to quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Show the frame
        cv2.imshow("Hand Gesture Recognition", frame)

        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # Save the current gesture
            gesture_name = input("Enter gesture name: ")
            gestures[gesture_name] = current_landmarks
            print(f"Gesture '{gesture_name}' saved!")
        elif key == ord('q'):  # Quit the application
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
