"""Streamlit entry point; adds this checkout's source package without installation."""

import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
runpy.run_module("tdwm.result_studio.app", run_name="__main__")
