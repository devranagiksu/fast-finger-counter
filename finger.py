import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands

# Camera setup - Optimized for performance
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

prev_time = 0
frame_skip = 3   # Process only every 3rd frame to save CPU
frame_counter = 0
finger_count = 0


with mp_hands.Hands(
    model_complexity=0, # Lowest complexity for maximum speed
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1) # Mirror effect
        frame_counter += 1

        # Logic: Only process frame if it's the Nth frame
        if frame_counter % frame_skip == 0:
            # Convert BGR to RGB for MediaPipe
            image_rgb = frame[:, :, ::-1]
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                # Indices for finger tips: Thumb(4), Index(8), Middle(12), Ring(16), Pinky(20)
                tip_ids = [4, 8, 12, 16, 20]
                finger_states = []

                for tip_id in tip_ids:
                    finger_tip = hand_landmarks.landmark[tip_id]
                    finger_mcp = hand_landmarks.landmark[tip_id - 1] # Joint below tip

                    if tip_id == 4: # Special logic for Thumb (horizontal movement)
                        finger_states.append(finger_tip.x < finger_mcp.x)
                    else: # Vertical logic for other fingers
                        finger_states.append(finger_tip.y < finger_mcp.y)

                finger_count = finger_states.count(True)

        # FPS Calculation
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time

        # Overlay Info
        cv2.putText(frame, f'Fingers: {finger_count}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        cv2.putText(frame, f'FPS: {int(fps)}', (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow("Fast Finger Counter", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
