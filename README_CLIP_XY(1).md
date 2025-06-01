
# CLIP-Based Object Detection and Robot Coordinate Mapping

This project demonstrates how to use OpenAI's CLIP model with a RealSense RGB camera to locate a physical object (e.g., red bolt) on a workspace and estimate its corresponding robot X and Y coordinates in millimeters.

---

## 🧠 Purpose

Automatically detect and localize a user-defined object using natural language (e.g., `"red blob"`) and estimate where it is in the robot's coordinate space, enabling pick-and-place or robotic interaction.

---

## 🖥️ Hardware & Tools

- **Intel RealSense Camera** (RGB stream)
- **Dobot Magician Lite** robot arm
- **OpenAI CLIP (ViT-B/32)** model
- **Python 3.10+**
- Libraries: `clip`, `torch`, `opencv-python`, `pyrealsense2`, `PIL`

---

## 📸 Camera to Robot Calibration

- Screen center: `(320, 240)` pixels → Robot: `(X=250 mm, Y=0 mm)`
- Conversion formulas:
  ```python
  robot_y = (pixel_x - 320) * 0.6818
  robot_x = 250 - (pixel_y - 240) * 0.625
  ```

---

## 🔍 Detection Algorithm

1. **Prompt** is encoded via CLIP (e.g., `"red blob"`)
2. **Full-frame sliding window scan** of the image using 128×128 patches
3. Each patch scored for similarity to the prompt
4. Highest scoring patch center is selected as the object location

---

## 📐 Coordinate Mapping

The center pixel of the best CLIP patch is converted into physical coordinates using calibration factors.

---

## 🧪 Trial Execution Flow

For each trial:
1. User places the object
2. Frame is captured from RealSense
3. CLIP-based scan determines best match
4. Pixel location is converted to robot X/Y
5. User enters actual known X/Y for comparison
6. Deviation is logged and image is annotated

---

## ✅ Outputs

- **Annotated image** saved to `/images_xy/`
- **Text log** saved to `deviation_log_xy.txt`
  - Includes trial #, pixel (x,y), estimated robot (X,Y), actual (X,Y), and deviations

---

## 📊 Sample Accuracy

- Y-axis (horizontal) detection: typically < 2 mm deviation
- X-axis (vertical) detection: typically ~30 mm deviation
- Detection is robust even under varied lighting and object types

---

## 📁 File List

- `full_scan_xy_detection.py`: Main detection and logging script
- `images_xy/`: Saved annotated images per trial
- `deviation_log_xy.txt`: Results of all trials

---

## 🔧 Future Enhancements

- Improve pixel-to-mm calibration via regression on more points
- Add support for Z-height estimation using RealSense depth
- Automatically move robot to predicted position for pick testing

---

## 🙌 Credits

Project by Daniel Christadoss and OpenAI's CLIP vision model.

