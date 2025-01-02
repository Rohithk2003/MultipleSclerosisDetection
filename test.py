import cv2
import numpy as np

# Load the pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# Function to apply filters to a detected face
def apply_filter(frame, face_coords, filter_type):
    x, y, w, h = face_coords
    face_roi = frame[y : y + h, x : x + w]

    if filter_type == "grayscale":
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2BGR)

    elif filter_type == "blur":
        face_roi = cv2.GaussianBlur(face_roi, (15, 15), 0)

    elif filter_type == "sepia":
        sepia_filter = np.array(
            [[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]]
        )
        face_roi = cv2.transform(face_roi, sepia_filter)
        face_roi = np.clip(face_roi, 0, 255).astype(np.uint8)

    frame[y : y + h, x : x + w] = face_roi


# Open webcam feed
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

current_filter = "none"

print(
    "Press 'g' for grayscale filter, 'b' for blur, 's' for sepia, 'n' for no filter, and 'q' to quit."
)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # Apply filter to each detected face
    for x, y, w, h in faces:
        if current_filter != "none":
            apply_filter(frame, (x, y, w, h), current_filter)

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the frame
    cv2.imshow("Face Filter", frame)

    # Key press handling
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("g"):
        current_filter = "grayscale"
    elif key == ord("b"):
        current_filter = "blur"
    elif key == ord("s"):
        current_filter = "sepia"
    elif key == ord("n"):
        current_filter = "none"

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
