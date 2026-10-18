from pathlib import Path
from PIL import Image

root = Path("dataset")
extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
bad_images = []

for path in root.rglob("*"):
    if not path.is_file() or path.suffix.lower() not in extensions:
        continue

    try:
        with Image.open(path) as img:
            img.verify()

        with Image.open(path) as img:
            img.load()

    except Exception as e:
        bad_images.append((path, str(e)))
        print(f"BAD IMAGE: {path}")
        print(f"ERROR: {e}\n")

print(f"\nScan complete. Bad images: {len(bad_images)}")
