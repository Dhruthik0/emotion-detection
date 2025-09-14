import cv2
from deepface import DeepFace

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Analyze frame for emotions
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # DeepFace returns dict, pick the first result
        if isinstance(result, list):
            result = result[0]

        # Get dominant emotion
        emotion = result['dominant_emotion']

        # Get face region (x, y, w, h)
        region = result['region']
        x, y, w, h = region['x'], region['y'], region['w'], region['h']

        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Put emotion label above the face
        cv2.putText(frame,
                    emotion,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

    except Exception as e:
        # In case no face detected
        print("No face detected:", e)

    # Show the video
    cv2.imshow("Facial Emotion Detection", frame)

    # Break on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()