import os
import datetime

def scan_repository(path="."):
    for root, dirs, files in os.walk(path):
        level = root.replace(path, "").count(os.sep)
        indent = " " * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 4 * (level + 1)
        for name in files:
            file_path = os.path.join(root, name)
            stats = os.stat(file_path)
            size_kb = stats.st_size / 1024
            last_modified = datetime.datetime.fromtimestamp(stats.st_mtime)
            ext = os.path.splitext(name)[1]
            print(f"{subindent}{name} | {ext} | {size_kb:.1f} KB | {last_modified}")

# Run from current repo path
scan_repository(".")
