"""Tracker layouts supported by the realtime MobilePose bridge.

Keep this module dependency-free.  Runtime layout detection must not import
the historical dataset/surface-IMU generation code.
"""

LAYOUTS = {
    "wrists_thighs_waist": {
        "labels": ["lw", "rw", "lt", "rt", "waist"],
    },
    "wrists_shanks_waist": {
        "labels": ["lw", "rw", "ls", "rs", "waist"],
    },
    "wrists_feet_waist": {
        "labels": ["lw", "rw", "lf", "rf", "waist"],
    },
    "upperarms_thighs_waist": {
        "labels": ["lu", "ru", "lt", "rt", "waist"],
    },
    "wrists_upperarms_waist": {
        "labels": ["lw", "rw", "lu", "ru", "waist"],
    },
    "legs_waist": {
        "labels": ["lt", "rt", "ls", "rs", "waist"],
    },
}
