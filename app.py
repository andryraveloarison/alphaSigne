
import cv2
import mediapipe as mp

# Initialize MediaPipe Hands and Drawing Utilities
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Initialize Webcam
cap = cv2.VideoCapture(0)

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

        # If hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks and connections on the hand
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get coordinates of specific landmarks (e.g., the tip of the index finger)
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, _ = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if id == 8:  # Index finger tip
                        cv2.circle(frame, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

        # Display the frame
        cv2.imshow("Hand Detection", frame)

        # Break on pressing 'ESC'
        if cv2.waitKey(1) & 0xFF == 27:
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
