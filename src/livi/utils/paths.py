from pathlib import Path
from livi.config import settings

ROOT = Path(__file__).resolve().parents[3]


class Paths:
    root = ROOT


p = Paths()
