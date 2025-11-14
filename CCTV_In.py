"""
cctv_integration.py
-------------------
Real-time CCTV / webcam integration for our
"Intelligent Surveillance System for Real-Time Weapon Detection" project.

- Loads the fine-tuned YOLOv8 model (best.pt)
- Connects to a CCTV stream (RTSP) or local webcam
- Runs real-time detection and overlays bounding boxes + labels
- Displays FPS on the video
"""

from ultralytics import YOLO
import cv2
import time

# ---------------------------------------------------
# CONFIGURE THESE TWO LINES FOR YOUR SETUP
# ---------------------------------------------------

# 1) Path to your trained weights (update if needed)
MODEL_PATH = "runs/weapon_yolov8/yolov8s_weapon/weights/best.pt"
# e.g. "runs/detect/train/weights/best.pt" depending on your training folder

# 2) CCTV / Webcam source:
#    - 0  -> laptop webcam
#    - "rtsp://user:pass@ip:554/Streaming/Channels/101" -> CCTV RTSP URL
VIDEO_SOURCE = 0  # change to your RTSP URL when you have it


def main():
    # Load YOLOv8 model
    print(f"[INFO] Loading model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # Open video stream
    print(f"[INFO] Opening video source: {VIDEO_SOURCE}")
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        print("[ERROR] Could not open video source. Check camera/RTSP URL.")
        return

    frame_count = 0
    t0 = time.time()

    print("[INFO] Press 'q' to quit the window.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Empty frame received, stopping.")
            break

        # Run YOLO inference on the current frame
        results = model.predict(
            source=frame,
            imgsz=640,
            conf=0.5,
            verbose=False
        )

        # YOLO returns a list of Results; take the first and plot detections
        annotated_frame = results[0].plot()  # BGR image with boxes/labels

        # FPS calculation
        frame_count += 1
        elapsed = time.time() - t0
        fps = frame_count / max(elapsed, 1e-6)

        # Put FPS text on frame
        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )

        # Show result
        cv2.imshow("Real-Time Weapon Detection - CCTV Integration", annotated_frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Stream closed. Total frames:", frame_count)


if __name__ == "__main__":
    main()
