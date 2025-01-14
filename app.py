import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Hands and Drawing Utilities
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Initialize Webcam
cap = cv2.VideoCapture(0)

# Memory for storing gestures
gestures = {}  # Format: {"Gesture Name": [[[x1, y1, z1], ...], [[x1, y1, z1], ...]]}
phrase = []  # List to store the words of the phrase

# Track last gesture and time
last_gesture = None
last_gesture_time = 0

# Function to normalize landmarks
def normalize_landmarks(landmarks):
    base_x, base_y, base_z = landmarks[0]  # Use wrist (landmark 0) as the base point
    normalized = []
    for x, y, z in landmarks:
        normalized.append([x - base_x, y - base_y, z - base_z])  # Center the landmarks
    # Calculate scaling factor (distance between wrist and middle finger tip)
    max_distance = max(np.linalg.norm(np.array(pt)) for pt in normalized)
    normalized = [[x / max_distance, y / max_distance, z / max_distance] for x, y, z in normalized]
    return normalized

# Function to calculate similarity between gestures
def is_matching_gesture(landmarks, saved_landmarks, threshold=0.07):
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
                    current_landmarks.append([lm.x * w, lm.y * h, lm.z * w])

                # Normalize the current landmarks
                normalized_landmarks = normalize_landmarks(current_landmarks)

                # Check for matching gestures
                matched_gesture = None
                for gesture_name, gesture_variants in gestures.items():
                    for saved_landmarks in gesture_variants:
                        if is_matching_gesture(normalized_landmarks, saved_landmarks):
                            matched_gesture = gesture_name
                            break
                    if matched_gesture:
                        break

                # Handle matched gesture
                current_time = time.time()
                if matched_gesture:
                    # Allow adding the same word if a certain time has passed
                    if matched_gesture != last_gesture or (current_time - last_gesture_time > 2):  # 2 seconds delay
                        phrase.append(matched_gesture)
                        last_gesture = matched_gesture
                        last_gesture_time = current_time

                    # Display the matched gesture
                    cv2.putText(frame, f"Gesture: {matched_gesture}", (450, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the constructed phrase with a black background
        phrase_text = "".join(phrase)
        text_size = cv2.getTextSize(phrase_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x, text_y = 10, 100  # Position of the text
        text_w, text_h = text_size[0], text_size[1]

        # Draw a black rectangle as the background
        cv2.rectangle(frame, (text_x - 5, text_y - text_h - 5), (text_x + text_w + 5, text_y + 5), (0, 0, 0), -1)

        # Display the phrase text on top of the rectangle
        cv2.putText(frame, phrase_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Instructions for memorizing gestures
        cv2.putText(frame, "Press 's' to save gesture, 'c' to clear, 'q' to quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Show the frame
        cv2.imshow("Hand Gesture Phrase Builder", frame)

        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # Save the current gesture
            gesture_name = input("Enter gesture name (word): ")
            if current_landmarks:
                normalized_gesture = normalize_landmarks(current_landmarks)
                if gesture_name in gestures:
                    gestures[gesture_name].append(normalized_gesture)
                else:
                    gestures[gesture_name] = [normalized_gesture]
                print(f"Gesture for '{gesture_name}' saved!")
        elif key == ord('c'):  # Clear the current phrase
            phrase = []
            print("Phrase cleared!")
        elif key == ord('q'):  # Quit the application
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
