import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for directory in (ROOT / "src", ROOT / "tools", ROOT / "tests"):
    dir_str = str(directory)
    if dir_str not in sys.path:
        sys.path.insert(0, dir_str)
