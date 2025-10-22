from flask import Flask, render_template, request, Response, jsonify
from ultralytics import YOLO
import cv2
import os
import uuid
import threading
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/output'

# Khởi tạo model YOLOv8
model = YOLO('yolov8n.pt')  # bạn có thể thay bằng yolov8s.pt nếu máy mạnh hơn

# Tạo thư mục nếu chưa có
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Biến toàn cục cho camera
camera_on = False
camera_thread = None
cap = None
current_count = 0


# ====== HÀM XỬ LÝ ẢNH ======
def process_image(image_path):
    results = model(image_path)
    detections = results[0].boxes
    count = sum(1 for box in detections if model.names[int(box.cls[0])] == 'person')
    output_img = results[0].plot()
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], os.path.basename(image_path))
    cv2.imwrite(output_path, output_img)
    return count, output_path


# ====== TRANG CHÍNH ======
@app.route('/')
def index():
    return render_template('index.html')


# ====== UPLOAD ẢNH / VIDEO ======
@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if not file:
        return render_template('index.html', error="Chưa chọn file!")

    filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # =========== XỬ LÝ ẢNH ===========
    if file.mimetype.startswith('image'):
        count, output_path = process_image(filepath)
        return render_template('index.html', uploaded_image=filepath, output_image=output_path, count=count)

    # =========== XỬ LÝ VIDEO ===========
    elif file.mimetype.startswith('video'):
        output_video = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        cap_video = cv2.VideoCapture(filepath)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, 20.0,
                              (int(cap_video.get(3)), int(cap_video.get(4))))

        total_frames = 0
        total_people_detected = 0

        while True:
            ret, frame = cap_video.read()
            if not ret:
                break

            total_frames += 1
            results = model(frame)
            detections = results[0].boxes
            count_in_frame = sum(1 for box in detections if model.names[int(box.cls[0])] == 'person')
            total_people_detected += count_in_frame

            annotated = results[0].plot()
            cv2.putText(annotated, f'People: {count_in_frame}', (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            out.write(annotated)

        cap_video.release()
        out.release()

        avg_people = round(total_people_detected / total_frames, 2) if total_frames > 0 else 0

        return render_template('index.html', uploaded_image=filepath,
                               output_video=output_video,
                               count=f"Trung bình {avg_people} người / frame")

    # =========== KHÔNG HỖ TRỢ ===========
    else:
        return render_template('index.html', error="Định dạng không được hỗ trợ!")


# ====== STREAM CAMERA ======
def generate_camera_stream():
    global cap, camera_on, current_count
    while camera_on:
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(0)
            time.sleep(1)

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        results = model(frame)
        detections = results[0].boxes
        current_count = sum(1 for box in detections if model.names[int(box.cls[0])] == 'person')

        annotated = results[0].plot()
        cv2.putText(annotated, f'People: {current_count}', (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        ret, buffer = cv2.imencode('.jpg', annotated)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        time.sleep(0.03)

    if cap and cap.isOpened():
        cap.release()
        cap = None


@app.route('/camera_feed')
def camera_feed():
    return Response(generate_camera_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ====== BẬT / TẮT CAMERA ======
@app.route('/toggle_camera', methods=['POST'])
def toggle_camera():
    global camera_on, camera_thread, cap
    data = request.get_json()

    if data['action'] == 'start':
        if not camera_on:
            camera_on = True
            camera_thread = threading.Thread(target=lambda: None)
            return jsonify(status='started')
    else:
        camera_on = False
        if cap and cap.isOpened():
            cap.release()
            cap = None
        return jsonify(status='stopped')


# ====== LẤY SỐ NGƯỜI HIỆN TẠI ======
@app.route('/current_count')
def get_current_count():
    return jsonify(count=current_count)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
