"""Replay a recorded raw-imu.json through the realtime adapter and model.

The recording contains the same SolarXR fields consumed by ``run.py``. This
reconstructs wire-equivalent values (notably accelerationG -> m/s^2), applies
the realtime heading/orientation adapter, runs the matching checkpoint, and
broadcasts the predicted pose to Unity on port 21200.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np
import torch

from mobileposer.no_head_layouts import LAYOUTS
from mobileposer.realtime.calibration import calibrate_heading
from mobileposer.realtime.infer import InferenceSession, load_model
from mobileposer.mobile_export import MobilePose
from mobileposer.realtime.layout import ResolvedLayout
from mobileposer.realtime.rotmat import batch_smpl_matrix_to_unity_quat_xyzw
from mobileposer.realtime.stream_out import PoseBroadcaster


STANDARD_GRAVITY = 9.80665
LABEL_TO_ROLE = {
    "lw": "LEFT_LOWER_ARM", "rw": "RIGHT_LOWER_ARM",
    "lu": "LEFT_UPPER_ARM", "ru": "RIGHT_UPPER_ARM",
    "lt": "LEFT_UPPER_LEG", "rt": "RIGHT_UPPER_LEG",
    "ls": "LEFT_LOWER_LEG", "rs": "RIGHT_LOWER_LEG",
    "lf": "LEFT_FOOT", "rf": "RIGHT_FOOT", "waist": "WAIST",
}


def _quat(value: dict) -> list[float]:
    return [float(value[k]) for k in ("x", "y", "z", "w")]


def _vec(value: dict) -> list[float]:
    return [float(value[k]) for k in ("x", "y", "z")]


def load_capture(path: Path, layout: str):
    package = json.loads(path.read_text(encoding="utf-8"))
    labels = LAYOUTS[layout]["labels"]
    roles = [LABEL_TO_ROLE[label] for label in labels]
    raw_frames, adjusted_frames, accel_frames, times = [], [], [], []
    sensor_ids = None

    for frame_index, frame in enumerate(package.get("frames") or []):
        by_role = {sensor.get("trackerRole"): sensor for sensor in frame.get("sensors", [])}
        sensors = [by_role.get(role) for role in roles]
        if any(sensor is None for sensor in sensors):
            raise ValueError(f"frame {frame_index} is missing one of roles {roles}")
        for sensor in sensors:
            required = ("online", "hasAcceleration", "hasOrientation", "hasRotationReferenceAdjusted")
            if not all(sensor.get(key, False) for key in required):
                raise ValueError(f"frame {frame_index}, {sensor.get('trackerRole')}: unavailable signal")
            if sensor.get("accelerationKind") != "linear":
                raise ValueError(f"frame {frame_index}, {sensor.get('trackerRole')}: expected linear acceleration")
        current_ids = [sensor.get("sensorId") for sensor in sensors]
        if sensor_ids is None:
            sensor_ids = current_ids
        elif current_ids != sensor_ids:
            raise ValueError(f"physical sensor identity changed at frame {frame_index}")

        raw_frames.append([_quat(sensor["orientation"]) for sensor in sensors])
        adjusted_frames.append([_quat(sensor["rotationReferenceAdjusted"]) for sensor in sensors])
        accel_frames.append([
            [component * STANDARD_GRAVITY for component in _vec(sensor["accelerationG"])]
            for sensor in sensors
        ])
        times.append(float(frame["timeSeconds"]))

    if not raw_frames:
        raise ValueError(f"no frames in {path}")
    times = np.asarray(times, dtype=np.float64)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("capture timestamps are invalid or non-monotonic")
    return (
        labels,
        np.asarray(raw_frames, dtype=np.float32),
        np.asarray(adjusted_frames, dtype=np.float32),
        np.asarray(accel_frames, dtype=np.float32),
        times,
        sensor_ids,
        float(package.get("sampleRate", 30.0)),
    )


async def replay(args: argparse.Namespace) -> None:
    labels, raw, adjusted, accel, times, sensor_ids, recorded_fps = load_capture(args.capture, args.layout)
    # Recorded actions do not necessarily begin with a dedicated calibration
    # gesture, so use the whole capture as the offline heading-fit window. The
    # fitted transform is still the exact same calibrate_heading implementation
    # used by run.py.
    calibration = calibrate_heading(
        labels,
        {label: raw[:, i] for i, label in enumerate(labels)},
        {label: adjusted[:, i] for i, label in enumerate(labels)},
    )
    for label in labels:
        diagnostic = calibration.diagnostics[label]
        print(f"{label}: sensor={sensor_ids[labels.index(label)]}, "
              f"yaw={diagnostic['yaw_deg']:+.1f}deg, "
              f"residual_p95={diagnostic['delta_residual_p95_deg']:.2f}deg")

    resolved = ResolvedLayout(
        name=args.layout,
        labels=labels,
        label_to_tracker_key={label: sensor_ids[i] for i, label in enumerate(labels)},
    )
    model = (MobilePose().load_combined(args.checkpoint) if args.checkpoint is not None
             else load_model(args.layout, args.checkpoint_root))
    session = InferenceSession(resolved, model, calibration)
    contact_model = None
    if args.include_contact:
        from mobileposer.realtime.inspect_foot_contact import load_foot_contact
        contact_path = (args.checkpoint if args.checkpoint is not None else
                        args.checkpoint_root / args.layout / "1" / "base_model.pth")
        contact_model = load_foot_contact(contact_path)
    quats = []
    contacts = []
    for frame in range(len(times)):
        pose = session.step(adjusted[frame], accel[frame]).numpy()
        quats.append(batch_smpl_matrix_to_unity_quat_xyzw(pose))
        if contact_model is not None and frame % max(1, args.contact_interval) == 0:
            with torch.inference_mode():
                window = torch.from_numpy(session.window._window[None])
                joints = session.model.joints(window)
                logits = contact_model(torch.cat((joints, window), dim=-1))[0, -1]
                contacts.append(torch.sigmoid(logits).numpy())
        elif contact_model is not None:
            contacts.append(contacts[-1] if contacts else np.zeros(2, dtype=np.float32))

    broadcaster = PoseBroadcaster()
    server = await broadcaster.serve(args.host, args.port)
    fps = args.fps or recorded_fps
    interval = 1.0 / fps
    print(f"Raw-IMU replay server listening on ws://{args.host}:{args.port}")
    print(f"layout={args.layout}, frames={len(quats)}, fps={fps:g}")
    try:
        while True:
            started = time.perf_counter()
            for frame_index, frame in enumerate(quats):
                contact = contacts[frame_index] if contacts else None
                await broadcaster.broadcast(frame_index, args.layout + "-raw-imu-replay", frame, contact)
                await asyncio.sleep(max(0.0, interval - (time.perf_counter() - started)))
                started = time.perf_counter()
            if not args.loop:
                break
    finally:
        server.close()
        await server.wait_closed()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path, help="version-3 *.raw-imu.json")
    parser.add_argument("--layout", choices=sorted(LAYOUTS), default="wrists_shanks_waist")
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints/no_head_5imu_surface"))
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="explicit combined checkpoint, e.g. results/ours_multi_finetune/layouts/wrists_shanks_waist/model_finetuned.pth")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=21200)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--include-contact", action="store_true",
                        help="also send footContact=[left,right] probabilities")
    parser.add_argument("--contact-interval", type=int, default=30,
                        help="recompute contact every N frames (default: 30)")
    asyncio.run(replay(parser.parse_args()))


if __name__ == "__main__":
    main()
