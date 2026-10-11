import sys
from pathlib import Path

EMOTE = Path(__file__).resolve().parents[2]
if str(EMOTE) not in sys.path:
    sys.path.insert(0, str(EMOTE))
