from flask import Flask, render_template, request
import cv2
import os
from ultralytics import YOLO

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load YOLO model (download once)
model = YOLO("yolov8n.pt")

# COCO classes YOLOv8 uses
weapon_classes = {"knife", "scissors"}  # Gun not in COCO v8n by default
person_class = "person"

def detect_threat(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return "⚠️ Could not open video"

    motion_values = []
    detected_objects = set()
    prev_frame = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 5 != 0:
            continue

        # Convert frame to RGB for YOLO (Ultralytics prefers RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MOTION DETECTION
        if prev_frame is not None:
            diff = cv2.absdiff(prev_frame, frame)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)
            motion = cv2.countNonZero(thresh)
            motion_values.append(motion)

        prev_frame = frame

        # YOLO OBJECT DETECTION
        results = model(rgb_frame)
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                detected_objects.add(label)

    cap.release()

    if len(motion_values) == 0:
        return "⚠️ Could not analyze video"

    avg_motion = sum(motion_values) / len(motion_values)

    # DECISION LOGIC
    if any(w in detected_objects for w in weapon_classes):
        return "🔫 Weapon Detected"

    if person_class in detected_objects and avg_motion > 3000:  # Adjust threshold
        return "🚨 Violence Detected"

    if avg_motion > 1500:
        return "⚠️ Suspicious Activity"

    return "✅ Normal Video"


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("video")
        if file:
            os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
            filename = file.filename
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            result = detect_threat(filepath)

            if "Weapon" in result or "Violence" in result:
                danger_score = 90
                explanation = "High motion + dangerous object detected → High risk"
            elif "Suspicious" in result:
                danger_score = 60
                explanation = "Unusual motion detected → Possibly suspicious activity"
            else:
                danger_score = 10
                explanation = "No abnormal activity detected → Safe"

            return render_template(
                "index.html",
                result=result,
                danger_score=danger_score,
                explanation=explanation,
                video_path=f"/{filepath.replace(os.sep, '/')}"
            )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)