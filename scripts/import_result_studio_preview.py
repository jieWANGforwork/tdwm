"""Import selected real reference clips using an existing SSH control connection."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tdwm.result_studio.import_preview import main

if __name__ == "__main__":
    main()
