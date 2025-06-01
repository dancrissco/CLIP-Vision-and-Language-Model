import cv2
from pathlib import Path

# --- Configuration ---
image_folder = Path("/home/dan/video/pincher/images_xy")  # Adjust if needed
image_paths = sorted(image_folder.glob("*.jpg")) + sorted(image_folder.glob("*.png"))

if not image_paths:
    print("[ERROR] No images found in folder.")
    exit()

print(f"[INFO] Found {len(image_paths)} images.")

# --- Loop through images ---
for path in image_paths:
    img = cv2.imread(str(path))
    if img is None:
        print(f"[WARNING] Could not read: {path.name}")
        continue

    cv2.imshow("Image Viewer", img)
    print(f"[Viewing] {path.name} - Press any key for next, or ESC to exit.")
    key = cv2.waitKey(0)

    if key == 27:  # ESC key
        break

cv2.destroyAllWindows()
