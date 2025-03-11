import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open the default camera
cap = cv2.VideoCapture(0)

# Get screen dimensions for full-screen canvas
screen_width = int(cap.get(3))
screen_height = int(cap.get(4))

# Initialize previous coordinates for drawing lines
prev_x, prev_y = None, None

# Create a blank full-screen canvas (white)
draw_canvas = 255 * np.ones((screen_height, screen_width, 3), dtype=np.uint8)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

cv2.namedWindow('Flipped Full-Screen Drawing', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Flipped Full-Screen Drawing', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get coordinates of index finger tip (landmark 8)
            index_finger_tip = hand_landmarks.landmark[8]
            x, y = int(index_finger_tip.x * screen_width), int(index_finger_tip.y * screen_height)

            # Draw a line from previous point to current point
            if prev_x is not None and prev_y is not None:
                cv2.line(draw_canvas, (prev_x, prev_y), (x, y), (255, 0, 0), 5)

            # Update previous coordinates
            prev_x, prev_y = x, y

    # Flip the canvas horizontally
    flipped_canvas = cv2.flip(draw_canvas, 1)

    # Display full-screen flipped drawing
    cv2.imshow('Flipped Full-Screen Drawing', flipped_canvas)

    # Clear canvas if 'c' is pressed
    if cv2.waitKey(1) & 0xFF == ord('c'):
        draw_canvas = 255 * np.ones((screen_height, screen_width, 3), dtype=np.uint8)
        prev_x, prev_y = None, None

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
