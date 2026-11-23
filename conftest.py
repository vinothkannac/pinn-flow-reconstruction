# Without this pytest could not find the pinn folder when I run it from the main folder.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
