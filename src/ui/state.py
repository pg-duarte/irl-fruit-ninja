# src/ui/state.py

from enum import Enum, auto


class AppState(Enum):
    MENU = auto()
    PLAYING = auto()
    PAUSED = auto()
    HIGHSCORES = auto()
    CREDITS = auto()
