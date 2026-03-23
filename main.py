from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
from ultralytics import YOLO
import os
import uuid

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = YOLO("best.pt")

@app.get("/")
def home():
    return {"message": "API running 🚀"}


# IMAGE DETECTION
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    
    filename = f"{uuid.uuid4()}.jpg"

    with open(filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = model(filename, save=True)

    output_path = os.path.join(
        results[0].save_dir,
        os.path.basename(filename)
    )

    return FileResponse(output_path, media_type="image/jpeg")


# VIDEO DETECTION (🔥 SIMPLE + WORKING)
@app.post("/detect_video")
async def detect_video(file: UploadFile = File(...)):

    import cv2

    input_path = f"input_{uuid.uuid4()}.mp4"

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    cap = cv2.VideoCapture(input_path)

    width = 640
    height = 480
    fps = 20

    output_path = f"output_{uuid.uuid4()}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (width, height))
        frame_count += 1

        # 🔥 ONLY FIRST 50 FRAMES (FAST)
        if frame_count > 50:
            break

        results = model(frame, conf=0.3)
        annotated = results[0].plot()

        out.write(annotated)

    cap.release()
    out.release()

    return FileResponse(output_path, media_type="video/mp4")