"""
main.py - Drone AI Tracker
Chalao: python main.py
"""

import cv2
import time
import argparse
from detector import DroneDetector
from tracker  import DroneTracker


def run(source=0, predict_seconds=3):
    detector = DroneDetector()
    tracker  = DroneTracker()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Camera nahi khula: {source}")
        return

    fps_cam = cap.get(cv2.CAP_PROP_FPS) or 30
    no_detect_frames = 0
    MAX_LOST = 30
    t_prev = time.time()

    print("[Main] Chal raha hai! Press 'q' to quit, 's' for screenshot")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Main] Camera band ho gaya.")
            break

        frame_count += 1
        detections = detector.detect(frame)

        best = None
        if detections:
            best = max(detections, key=lambda d: d["conf"])
            no_detect_frames = 0

        if best:
            state = tracker.update(best["cx"], best["cy"])
        elif tracker.initialized and no_detect_frames < MAX_LOST:
            state = tracker.predict_only()
            no_detect_frames += 1
        else:
            state = None

        intercept = None
        if state:
            intercept = tracker.predict_future(
                seconds_ahead=predict_seconds, fps=fps_cam)

        frame = detector.draw(frame, detections)

        if state:
            frame = tracker.draw(frame, state, intercept)

        t_now  = time.time()
        fps    = 1.0 / max(t_now - t_prev, 1e-6)
        t_prev = t_now

        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 1)

        status = "TARGET LOCKED" if best else (
                 "TRACKING..." if state else "SEARCHING...")
        color  = (0,255,0) if best else ((0,200,255) if state else (100,100,100))
        cv2.putText(frame, status,
                    (10, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

        if intercept:
            cv2.putText(frame,
                        f"Intercept ({predict_seconds}s): {intercept}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1)

        cv2.imshow("Drone AI System", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f"screenshot_{frame_count}.jpg"
            cv2.imwrite(fname, frame)
            print(f"[Save] {fname}")

    cap.release()
    cv2.destroyAllWindows()
    print("[Main] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",  default=0,   help="Camera index ya video file")
    parser.add_argument("--predict", type=int, default=3, help="Seconds ahead")
    args = parser.parse_args()
    run(source=args.source, predict_seconds=args.predict)
