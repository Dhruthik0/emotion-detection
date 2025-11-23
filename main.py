# import cv2
# from deepface import DeepFace

# # Start webcam
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     try:
#         # Analyze frame for emotions
#         result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

#         # DeepFace returns dict, pick the first result
#         if isinstance(result, list):
#             result = result[0]

#         # Get dominant emotion
#         emotion = result['dominant_emotion']

#         # Get face region (x, y, w, h)
#         region = result['region']
#         x, y, w, h = region['x'], region['y'], region['w'], region['h']

#         # Draw rectangle around face
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#         # Put emotion label above the face
#         cv2.putText(frame,
#                     emotion,
#                     (x, y - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     1,
#                     (0, 255, 0),
#                     2)

#     except Exception as e:
#         # In case no face detected
#         print("No face detected:", e)

#     # Show the video
#     cv2.imshow("Facial Emotion Detection", frame)

#     # Break on 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# main.py
# main.py
import os
import io
from typing import Dict, Any

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from deepface import DeepFace

app = FastAPI()
templates = Jinja2Templates(directory="templates")
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def read_imagefile_to_cv2(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts multipart/form-data with key 'file' (an image blob).
    Returns JSON: {dominant_emotion, emotions: {...}, region: {x,y,w,h}}
    """
    try:
        contents = await file.read()
        img = read_imagefile_to_cv2(contents)
        if img is None:
            return JSONResponse({"error": "Could not decode image"}, status_code=400)

        # Convert BGR->RGB for DeepFace
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # analyze; enforce_detection=False avoids hard failure when no face, set True for stricter detection
        result = DeepFace.analyze(rgb, actions=["emotion"], enforce_detection=False)

        # DeepFace may return a list for multiple faces; pick the first result
        if isinstance(result, list):
            result = result[0]

        emotion = result.get("dominant_emotion", "")
        emotions = result.get("emotion", {})
        region = result.get("region", {}) or {}
        region = {k: int(region.get(k, 0)) for k in ("x", "y", "w", "h")}

        response: Dict[str, Any] = {
            "dominant_emotion": emotion,
            "emotions": emotions,
            "region": region,
        }
        return JSONResponse(response)

    except Exception as e:
        print("Error in /predict:", str(e))
        return JSONResponse({"error": str(e)}, status_code=500)
