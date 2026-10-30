import cv2

# Try index 0, 1, or 2 depending on your connected webcams/capture cards
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Use CAP_DSHOW on Windows for fast initialization

if not cap.isOpened():
    print("Error: Could not open SJCAM stream. Trying index 0...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Set resolution (e.g., 850x478 or 1280x720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 850)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 478)

print("Press 'q' to exit the test stream.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Get actual running properties
    h, w, _ = frame.shape
    cv2.putText(
        frame, 
        f"Resolution: {w}x{h}", 
        (20, 40), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.7, 
        (0, 255, 0), 
        2
    )

    cv2.imshow("SJCAM Beta Test Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()