"""Cloud entrypoint for Streamlit Community Cloud and similar hosts."""

from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
APP = SRC / "seuphyx" / "core" / "oil" / "app.py"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

runpy.run_path(str(APP), run_name="__main__")
