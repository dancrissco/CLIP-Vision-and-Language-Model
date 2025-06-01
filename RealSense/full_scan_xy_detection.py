
import cv2
import torch
import clip
import numpy as np
import pyrealsense2 as rs
from PIL import Image as PILImage
from pathlib import Path

# --- Configuration ---
prompt = "red blob"
num_trials = int(input("Enter number of trials: "))
image_dir = Path("images_xy")
image_dir.mkdir(exist_ok=True)
log_path = Path("deviation_log_xy.txt")

# CLIP setup
device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
text_tokens = clip.tokenize([prompt]).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_tokens).detach()

# RealSense setup
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Pixel to robot coordinate mapping
def pixel_x_to_robot_y(x_px):
    return (x_px - 320) * 0.6818

def pixel_y_to_robot_x(y_px):
    return 250 - (y_px - 240) * 0.625

def detect_best_pixel(image_np):
    best_score = -np.inf
    best_center = (image_np.shape[1] // 2, image_np.shape[0] // 2)

    with torch.no_grad():
        for y in range(0, image_np.shape[0] - 128, 32):
            for x in range(0, image_np.shape[1] - 128, 32):
                patch = image_np[y:y+128, x:x+128]
                pil_patch = PILImage.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
                img_tensor = preprocess(pil_patch).unsqueeze(0).to(device)
                score = (model.encode_image(img_tensor) @ text_features.T).item()
                if score > best_score:
                    best_score = score
                    best_center = (x + 64, y + 64)

    return best_center

# --- Trial loop ---
log_lines = []
x_devs = []
y_devs = []

print("\n🔁 Starting trials...\n")
log_lines.append("Trial | Pixel (X,Y) | Robot X (mm) | Robot Y (mm) | Act X | Act Y | Dev X | Dev Y")
log_lines.append("-" * 80)

for i in range(1, num_trials + 1):
    input(f"➡️ Position object for Trial {i} and press Enter...")

    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        print("[ERROR] No frame received.")
        continue
    img = np.asanyarray(color_frame.get_data())

    best_center = detect_best_pixel(img)
    pixel_x, pixel_y = best_center
    est_y = pixel_x_to_robot_y(pixel_x)
    est_x = pixel_y_to_robot_x(pixel_y)

    actual_x = float(input(f"Enter actual X position in mm for Trial {i}: "))
    actual_y = float(input(f"Enter actual Y position in mm for Trial {i}: "))
    dev_x = round(abs(est_x - actual_x), 2)
    dev_y = round(abs(est_y - actual_y), 2)
    x_devs.append(dev_x)
    y_devs.append(dev_y)

    # Annotate and save image
    annotated = img.copy()
    cv2.circle(annotated, best_center, 10, (0, 255, 0), -1)
    cv2.putText(annotated, f"X={est_x:.1f}, Y={est_y:.1f}", (pixel_x+10, pixel_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    filename = image_dir / f"trial_xy_{i:02d}.jpg"
    cv2.imwrite(str(filename), annotated)

    # Log result
    log_line = f"{i:<5} | ({pixel_x},{pixel_y}) | {est_x:>10.2f} | {est_y:>10.2f} | {actual_x:>5.1f} | {actual_y:>5.1f} | {dev_x:>6.2f} | {dev_y:>6.2f}"
    log_lines.append(log_line)
    print(log_line)

# --- Summary ---
log_lines.append("-" * 80)
log_lines.append(f"Avg Dev X: {np.mean(x_devs):.2f} mm, Y: {np.mean(y_devs):.2f} mm")
log_lines.append(f"Var Dev X: {np.var(x_devs):.2f}, Y: {np.var(y_devs):.2f}")

print("\n📊 Summary:")
for line in log_lines[-3:]:
    print(line)

with open(log_path, "w") as f:
    f.write("\n".join(log_lines))

print(f"\n✅ Log saved to {log_path.resolve()}")
pipeline.stop()
