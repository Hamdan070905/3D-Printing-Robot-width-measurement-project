import cv2

# Skip index 0 (laptop webcam) and test indices 1 and 2
for i in [1, 2]:
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Testing Camera Index {i}...")
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"Index {i} opened but failed to read frames.")
                break
                
            # Show the video feed so you can see if it's the SJCAM
            cv2.imshow(f"Testing Index {i} - Press 'q' to select this camera", frame)
            
            # Press 'q' when you see your SJCAM feed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(f"\n-> CONFIRMED: Your SJCAM is on Index {i}!")
                cap.release()
                cv2.destroyAllWindows()
                exit()
        cap.release()
    else:
        print(f"Index {i} is not available.")

cv2.destroyAllWindows()