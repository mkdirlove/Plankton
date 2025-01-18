from flask import Flask, Response, render_template
import torch
import cv2
from pathlib import Path

app = Flask(__name__)

# Load YOLOv5 model
def load_model():
    weights_path = Path(r"runs\train\exp11\weights\best.pt")
    return torch.hub.load('ultralytics/yolov5', 'custom', path=str(weights_path))

yolo_model = load_model()
print("YOLOv5 model loaded.")

# Video stream generator
def video_stream():
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Error: Unable to access the webcam.")
        return

    while True:
        ret, frame = capture.read()
        if not ret:
            print("Error: Failed to read frame from webcam.")
            break

        # Perform YOLOv5 detection
        results = yolo_model(frame)
        detected_frame = results.render()[0]  # Render detection results on the frame

        # Encode the frame in JPEG format for streaming
        _, buffer = cv2.imencode('.jpg', detected_frame)
        frame_bytes = buffer.tobytes()

        # Yield frame in HTTP streaming format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    capture.release()

@app.route('/')
def index():
    # Serve the main HTML page
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Serve the video feed
    return Response(video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
