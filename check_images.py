
from pathlib import Path
from collections import Counter, defaultdict
import hashlib

ROOT = Path("data")

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

images = [
    p for p in ROOT.rglob("*")
    if p.is_file() and p.suffix.lower() in extensions
]

print("TOTAL IMAGE FILES:", len(images))
print("\nIMAGE COUNTS BY PARENT FOLDER:")

counts = Counter(str(p.parent) for p in images)
for folder, count in sorted(counts.items()):
    print(f"{count:5}  {folder}")

hash_to_paths = defaultdict(list)

for path in images:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    hash_to_paths[digest].append(path)

duplicate_groups = [
    paths for paths in hash_to_paths.values()
    if len(paths) > 1
]

print("\nUNIQUE FILE CONTENTS:", len(hash_to_paths))
print("EXACT DUPLICATE GROUPS:", len(duplicate_groups))
print(
    "FILES IN DUPLICATE GROUPS:",
    sum(len(group) for group in duplicate_groups)
)

for group in duplicate_groups[:10]:
    print("\nDuplicate group:")
    for path in group:
        print(" ", path)
