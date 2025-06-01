import cv2
import pyrealsense2 as rs
import numpy as np

# --- RealSense Setup ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

print("[INFO] Starting RealSense stream...")
pipeline.start(config)

try:
    while True:
        # Get frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert to numpy image
        image = np.asanyarray(color_frame.get_data())

        # Get dimensions
        h, w, _ = image.shape
        center_x = w // 2
        center_y = h // 2

        # Draw centerlines
        cv2.line(image, (center_x, 0), (center_x, h), (0, 255, 0), 2)  # Vertical: Robot Y = 0
        cv2.line(image, (0, center_y), (w, center_y), (255, 0, 0), 2)  # Horizontal: Screen center

        # Label
        cv2.putText(image, "Robot Y = 0", (center_x + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(image, "Screen center", (10, center_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Show image
        cv2.imshow("RealSense Crosshair View", image)

        if cv2.waitKey(1) == 27:
            break  # ESC to quit

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("[INFO] RealSense stream stopped.")
