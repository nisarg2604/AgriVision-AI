
from pathlib import Path

root = Path("data")

print("Checking directories under:", root.resolve())

for path in sorted(root.rglob("*")):
    if path.is_dir():
        if path.name.lower() in {"train", "val", "valid", "validation", "test"}:
            print("\nPossible split folder:", path)

            for class_folder in sorted(path.iterdir()):
                if class_folder.is_dir():
                    count = sum(
                        1 for p in class_folder.iterdir()
                        if p.is_file() and p.suffix.lower()
                        in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
                    )
                    print(f"  {class_folder.name}: {count} images")
