"""Authoritative constants for the current five-IMU realtime product path."""

REALTIME_LAYOUT = "wrists_shanks_waist"
REALTIME_LABELS = ("lw", "rw", "ls", "rs", "waist")
WINDOW = 45
OUTPUT_INDEX = 40
OUTPUT_LAG_FRAMES = WINDOW - 1 - OUTPUT_INDEX
ACCELERATION_SCALE = 30.0
PARENTS = (
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21,
)
