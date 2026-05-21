"""
tracker.py
----------
Kalman Filter se drone ko continuously track karta hai.
Position ke saath velocity bhi estimate karta hai — yahi
trajectory prediction ke liye zaroori hai.
"""

import numpy as np
from filterpy.kalman import KalmanFilter


class DroneTracker:
    def __init__(self):
        """
        State vector: [x, y, vx, vy]
          x, y   = position (pixels)
          vx, vy = velocity (pixels/frame)
        """
        self.kf = KalmanFilter(dim_x=4, dim_z=2)

        # State transition matrix (constant velocity model)
        # x_new = x + vx*dt, y_new = y + vy*dt
        dt = 1.0
        self.kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1,  0, dt],
            [0, 0,  1,  0],
            [0, 0,  0,  1]
        ], dtype=float)

        # Measurement matrix — hum sirf x,y observe karte hain
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=float)

        # Measurement noise covariance
        self.kf.R = np.eye(2) * 10

        # Process noise covariance
        self.kf.Q = np.eye(4) * 0.1

        # Initial covariance
        self.kf.P = np.eye(4) * 500

        self.initialized = False
        self.history = []   # last N positions store karte hain
        self.max_history = 30

    def init(self, cx, cy):
        """Pehli baar target mila — state initialize karo."""
        self.kf.x = np.array([[cx], [cy], [0], [0]], dtype=float)
        self.initialized = True
        self.history = [(cx, cy)]

    def update(self, cx, cy):
        """Naya detection mila — filter update karo."""
        if not self.initialized:
            self.init(cx, cy)
            return self.get_state()

        self.kf.predict()
        self.kf.update(np.array([[cx], [cy]], dtype=float))

        state = self.get_state()
        self.history.append((state["x"], state["y"]))
        if len(self.history) > self.max_history:
            self.history.pop(0)

        return state

    def predict_only(self):
        """Jab koi detection na aaye — sirf predict karo."""
        if not self.initialized:
            return None
        self.kf.predict()
        return self.get_state()

    def get_state(self):
        """Current state return karo."""
        x  = float(self.kf.x[0])
        y  = float(self.kf.x[1])
        vx = float(self.kf.x[2])
        vy = float(self.kf.x[3])
        speed = np.sqrt(vx**2 + vy**2)
        return {"x": x, "y": y, "vx": vx, "vy": vy, "speed": speed}

    def predict_future(self, seconds_ahead=3, fps=30):
        """
        'seconds_ahead' seconds baad drone kahan hoga?
        Returns: (future_x, future_y)
        """
        if not self.initialized:
            return None

        frames = int(seconds_ahead * fps)
        state = self.get_state()
        fx = state["x"] + state["vx"] * frames
        fy = state["y"] + state["vy"] * frames
        return (int(fx), int(fy))

    def draw(self, frame, state, intercept_point=None):
        """
        Track history, velocity arrow, aur intercept point draw karo.
        """
        import cv2

        # Trail (history path)
        for i in range(1, len(self.history)):
            alpha = i / len(self.history)
            color = (0, int(255 * alpha), int(100 * (1 - alpha)))
            cv2.line(frame,
                     (int(self.history[i-1][0]), int(self.history[i-1][1])),
                     (int(self.history[i][0]),   int(self.history[i][1])),
                     color, 2)

        # Current position
        cx, cy = int(state["x"]), int(state["y"])
        cv2.circle(frame, (cx, cy), 8, (0, 255, 0), 2)

        # Velocity arrow
        vx, vy = state["vx"] * 5, state["vy"] * 5  # scale for visibility
        cv2.arrowedLine(frame,
                        (cx, cy),
                        (int(cx + vx), int(cy + vy)),
                        (255, 255, 0), 2, tipLength=0.3)

        # Speed text
        cv2.putText(frame,
                    f"Speed: {state['speed']:.1f} px/f",
                    (cx + 12, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

        # Intercept point
        if intercept_point:
            ix, iy = intercept_point
            cv2.drawMarker(frame, (ix, iy), (0, 0, 255),
                           cv2.MARKER_CROSS, 20, 2)
            cv2.circle(frame, (ix, iy), 12, (0, 0, 255), 1)
            cv2.putText(frame, "INTERCEPT",
                        (ix + 14, iy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            # Dashed line: current → intercept
            cv2.line(frame, (cx, cy), (ix, iy), (100, 100, 255), 1,
                     cv2.LINE_AA)

        return frame
