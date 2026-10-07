"""Compare offline ours preprocessing with the realtime SolarXR path."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

from mobileposer.no_head_layouts import LAYOUTS
from mobileposer.realtime.calibration import bone_orientation, calibrate_heading
from mobileposer.realtime.infer import build_feature
from mobileposer.test_ours_surface import prepare

_ROLES = {
    "lw": "LEFT_LOWER_ARM", "rw": "RIGHT_LOWER_ARM",
    "ls": "LEFT_LOWER_LEG", "rs": "RIGHT_LOWER_LEG", "waist": "WAIST",
}


def _quat(sample, key):
    value = sample[key]
    return np.asarray([value[k] for k in ("x", "y", "z", "w")], dtype=np.float32)


def compare_capture(capture: str | Path, layout: str = "wrists_shanks_waist", calibration_frames: int = 150):
    capture = Path(capture).resolve()
    manifest = json.loads((capture / "manifest.json").read_text(encoding="utf-8"))
    raw = json.loads((capture / manifest["rawImuJsonFile"]).read_text(encoding="utf-8"))
    labels = list(LAYOUTS[layout]["labels"])
    roles = [_ROLES[label] for label in labels]
    frames = raw["frames"]
    source_times = np.asarray([float(frame.get("timeSeconds", index / 30.0)) for index, frame in enumerate(frames)], dtype=np.float64)
    raw_q = {label: [] for label in labels}
    adjusted_q = {label: [] for label in labels}
    acceleration = {label: [] for label in labels}
    for frame in frames:
        by_role = {item.get("trackerRole"): item for item in frame.get("sensors", [])}
        if any(role not in by_role for role in roles):
            continue
        for label, role in zip(labels, roles):
            item = by_role[role]
            raw_q[label].append(_quat(item, "orientation"))
            adjusted_q[label].append(_quat(item, "rotationReferenceAdjusted"))
            a = np.asarray([item["accelerationG"][k] for k in ("x", "y", "z")], dtype=np.float32) * 9.80665
            if item.get("accelerationKind") == "raw":
                a = Rotation.from_quat(raw_q[label][-1]).as_matrix() @ a
            acceleration[label].append(a)
    count = min(len(v) for v in raw_q.values())
    if count < 2:
        raise ValueError("capture has fewer than two complete frames for the selected layout")
    raw_q = {label: np.asarray(values[:count]) for label, values in raw_q.items()}
    adjusted_q = {label: np.asarray(values[:count]) for label, values in adjusted_q.items()}
    acceleration = {label: np.asarray(values[:count]) for label, values in acceleration.items()}

    # Keep the intermediate output in a normal project directory.  Some
    # managed Windows Python installations deny nested writes in temp dirs.
    temp = Path("recordings/feature_compare_temp")
    temp.mkdir(parents=True, exist_ok=True)
    report = prepare(capture, temp)
    entry = report["layouts"].get(layout)
    if not entry or entry.get("status") != "prepared":
        raise ValueError(f"offline prepare did not produce layout {layout}: {entry}")
    offline_data = torch.load(entry["input"], map_location="cpu", weights_only=False)
    offline = torch.cat(((offline_data["acc"][0] / 30).flatten(1), offline_data["ori"][0].flatten(1)), dim=1).numpy()
    target_times = offline_data["timestamps"].numpy().astype(np.float64)

    heading = calibrate_heading(
        labels,
        {label: values[:calibration_frames] for label, values in raw_q.items()},
        {label: values[:calibration_frames] for label, values in adjusted_q.items()},
    )
    # Use the exact timestamps used by offline prepare: acceleration is
    # linearly interpolated and orientation uses quaternion Slerp.  This
    # removes acquisition-rate jitter from the comparison itself.
    source_acc = np.stack([acceleration[label] for label in labels], axis=1)
    basis = torch.tensor([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
    aligned_source_acc = torch.stack([
        (basis @ (heading.heading[label_index] @ torch.from_numpy(source_acc[:, label_index]).T)).T
        for label_index in range(len(labels))
    ], dim=1).numpy()
    realtime_acc = [np.stack([
        np.interp(target_times, source_times[:count], aligned_source_acc[:, label_index, axis])
        for axis in range(3)
    ], axis=-1) for label_index in range(len(labels))]
    realtime_ori = []
    for label in labels:
        rotations = Rotation.from_quat(adjusted_q[label])
        realtime_ori.append(
            Slerp(source_times[:count], rotations)(target_times).as_matrix()
        )
    realtime = np.stack([
        build_feature(
            bone_orientation(
                np.stack([
                    Rotation.from_matrix(realtime_ori[label_index][frame]).as_quat()
                    for label_index in range(len(labels))
                ]), labels,
            ),
            torch.from_numpy(np.stack([realtime_acc[label_index][frame] for label_index in range(len(labels))])),
        )
        for frame in range(len(target_times))
    ], axis=0)
    count = min(len(offline), len(realtime))
    delta = realtime[:count] - offline[:count]
    per_label = {}
    for index, label in enumerate(labels):
        values = delta[:, index * 3:index * 3 + 3]
        values = np.concatenate((values, delta[:, 15 + index * 9:15 + (index + 1) * 9]), axis=1)
        per_label[label] = {"maxAbs": float(np.abs(values).max()), "rmse": float(np.sqrt(np.mean(values ** 2)))}
    return {
        "capture": str(capture), "layout": layout, "framesCompared": count,
        "calibrationFrames": calibration_frames,
        "featureMaxAbs": float(np.abs(delta).max()),
        "featureRmse": float(np.sqrt(np.mean(delta ** 2))),
        "accelerationMaxAbs": float(np.abs(delta[:, :15]).max()),
        "accelerationRmse": float(np.sqrt(np.mean(delta[:, :15] ** 2))),
        "orientationMaxAbs": float(np.abs(delta[:, 15:]).max()),
        "orientationRmse": float(np.sqrt(np.mean(delta[:, 15:] ** 2))),
        "perTracker": per_label,
        "headingDiagnostics": heading.diagnostics,
    }
