

import math
import os
import cv2
import sqlite3
import threading
from flask import Flask, request, jsonify, render_template, send_file
from ultralytics import YOLO
from queue import Queue

app = Flask(__name__)

# Paths for uploads and results
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Path to custom dataset YAML file
DATA_YAML_PATH = '/home/wens/Company/VDS/Project/uleog_data.yaml'

# YOLO model initialization
model_path = '/home/wens/Company/VDS/Project/uleog_20241120.pt'
model = YOLO(model_path)
model.to('cpu')
model.overrides['data'] = DATA_YAML_PATH

# Global variables
video_queue = Queue()
currently_processing = None
stop_event = threading.Event()  # Event to stop the loop
processing_thread = None

# SQLite database file
DB_FILE = "labels.db"


def init_db():
    """
    Initialize the SQLite database and create the required table if it doesn't exist.
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Create table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            class_name TEXT NOT NULL,
            label_name TEXT NOT NULL,
            accuracy REAL NOT NULL,
            speed REAL
        )
    ''')

    conn.commit()
    conn.close()


def save_label_to_db(class_name, label_name, accuracy, speed=None):
    """
    Save a detected label and speed to the SQLite database.
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO detections (class_name, label_name, accuracy, speed)
        VALUES (?, ?, ?, ?)
    ''', (class_name, label_name, accuracy, speed))

    conn.commit()
    conn.close()


def calculate_speed(position1, position2, fps, scale=0.1):
    """
    Calculate the speed of an object in kilometers per hour (km/h).
    Args:
        position1 (tuple): (x, y) coordinates in the first frame.
        position2 (tuple): (x, y) coordinates in the second frame.
        fps (int): Frames per second of the video.
        scale (float): Scaling factor to convert pixels to meters.
    Returns:
        float: Speed in km/h.
    """
    # Calculate distance in pixels
    distance_pixels = math.sqrt((position2[0] - position1[0])**2 + (position2[1] - position1[1])**2)

    # Convert pixel distance to meters
    distance_meters = distance_pixels * scale

    # Calculate time between frames
    time_seconds = 1 / fps if fps > 0 else 1  # Avoid division by zero

    # Calculate speed in meters per second
    speed_mps = distance_meters / time_seconds

    # Convert speed to km/h
    speed_kmh = speed_mps * 3.6

    # Debug logs
    print(f"Debug -> distance_pixels: {distance_pixels:.2f}, distance_meters: {distance_meters:.2f}, "
          f"time_seconds: {time_seconds:.4f}, speed_mps: {speed_mps:.2f}, speed_kmh: {speed_kmh:.2f}")

    return speed_kmh



@app.route('/')
def index():
    return render_template('inde.html')


@app.route('/upload_and_detect', methods=['POST'])
def upload_and_detect():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video = request.files['video']
    if video.filename == '':
        return jsonify({'error': 'No video file provided'}), 400

    video_path = os.path.join(UPLOAD_FOLDER, video.filename)
    video.save(video_path)
    video_queue.put(video_path)
    print(f"Video uploaded and added to queue: {video_path}")

    # Start the processing thread if not already running
    global processing_thread
    if not processing_thread or not processing_thread.is_alive():
        print("Starting video processing thread.")
        processing_thread = threading.Thread(target=process_videos, daemon=True)
        processing_thread.start()

    return jsonify({'success': True, 'message': 'Video uploaded and detection started!'})


def process_videos():
    """
    Continuously process videos from the queue in the main thread.
    """
    global currently_processing

    while not stop_event.is_set():
        if currently_processing is None and not video_queue.empty():
            video_path = video_queue.get()
            currently_processing = video_path
            result_path = os.path.join(RESULTS_FOLDER, f'detected_{os.path.basename(video_path)}')
            process_video(video_path, result_path)
            currently_processing = None

        # Sleep briefly to prevent high CPU usage
        stop_event.wait(0.1)


def process_video(video_path, result_path):
    """
    Process the uploaded video, calculate speed, save detected video, 
    and save labels with accuracy > 0.70 to the database.
    """
    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width, height = 1280, 720
    out = cv2.VideoWriter(result_path, fourcc, fps, (width, height))

    cv2.namedWindow('Detection Window', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Detection Window', width, height)

    # Dictionary to track vehicle positions across frames
    vehicle_positions = {}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (width, height))
            results = model(frame)

            # Filter and process results for accuracy > 0.70
            filtered_boxes = [
                box for result in results if result.boxes 
                for box in result.boxes if float(box.conf.item()) > 0.70
            ]

            for box in filtered_boxes:
                class_index = int(box.cls.item())
                confidence = float(box.conf.item())
                bbox = box.xyxy[0].cpu().numpy()  # Get bounding box coordinates
                center_x, center_y = int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)

                class_name = model.names[class_index]
                label_name = f"Label-{class_index}"

                # Calculate speed
                if class_name not in vehicle_positions:
                    vehicle_positions[class_name] = (center_x, center_y)
                    speed_kmh = 0.0
                else:
                    speed_kmh = calculate_speed(vehicle_positions[class_name], (center_x, center_y), fps, scale=0.05)
                    vehicle_positions[class_name] = (center_x, center_y)

                # Save to database
                save_label_to_db(class_name, label_name, confidence, speed_kmh)

                # Annotate frame with speed
                cv2.putText(
                    frame,
                    f"{class_name} {confidence:.2f} Speed: {speed_kmh:.2f} km/h",
                    (int(bbox[0]), int(bbox[1] - 10)),  # Position above the bounding box
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,  # Font scale
                    (0, 255, 0),  # Green color
                    2,  # Thickness
                    cv2.LINE_AA
                )

                # Draw the bounding box
                cv2.rectangle(
                    frame,
                    (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])),
                    (255, 0, 0),  # Blue color
                    2  # Thickness
                )

                # Print to console for verification
                print(f"Saved to DB -> Class: {class_name}, Speed: {speed_kmh:.2f} km/h")

            out.write(frame)
            cv2.imshow('Detection Window', frame)

            if cv2.waitKey(1) & 0xFF == ord('q') or not cv2.getWindowProperty('Detection Window', cv2.WND_PROP_VISIBLE):
                print("Detection window closed manually.")
                break
    except Exception as e:
        print(f"Error during detection: {e}")
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Processing completed for video: {video_path}")


@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(RESULTS_FOLDER, filename), as_attachment=True)


if __name__ == '__main__':
    init_db()  # Initialize the database
    try:
        app.run(debug=False)  # Disable Flask's debug mode for threading
    finally:
        stop_event.set()
        if processing_thread:
            processing_thread.join()
