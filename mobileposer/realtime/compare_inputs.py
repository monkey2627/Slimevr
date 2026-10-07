"""Compare a realtime JSONL capture with processed ours input tensors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mobileposer.realtime.recording import load_recording


def _stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "shape": list(values.shape),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "rms": np.sqrt(np.mean(values * values, axis=0)).tolist(),
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
    }


def compare(recording: Path, ours: Path, sequence: int) -> dict:
    metadata, rows = load_recording(recording)
    frames = [row for row in rows if row.get("type") in ("frame", "inferenceFrame")]
    if not frames:
        raise ValueError("recording contains no inference frames")
    live_acc = np.asarray([row["alignedAccelerationMps2"] for row in frames], dtype=np.float32)
    live_ori = np.asarray([row["calibratedOrientationMatrices"] for row in frames], dtype=np.float32)
    live_time = np.asarray([
        row.get("receiveMonotonicSeconds", row.get("hostMonotonicSeconds"))
        for row in frames
    ], dtype=np.float64)
    if np.isnan(live_time).any():
        raise ValueError("recording inference frames do not contain receive timestamps")
    data = torch.load(ours, map_location="cpu", weights_only=False)
    ours_acc = torch.as_tensor(data["acc"][sequence]).float().numpy()[:, :5]
    ours_ori = torch.as_tensor(data["ori"][sequence]).float().numpy()[:, :5]
    if list(metadata["labels"]) != list(data["layout_labels"]):
        raise ValueError(f"label mismatch: live={metadata['labels']}, ours={data['layout_labels']}")
    dt = np.diff(live_time)
    return {
        "labels": metadata["labels"],
        "liveFrames": len(live_acc),
        "oursFrames": len(ours_acc),
        "liveFrameIntervalSeconds": _stats(dt[:, None]) if len(dt) else None,
        "liveAccelerationMps2": _stats(live_acc),
        "oursAccelerationMps2": _stats(ours_acc),
        "liveAccelerationScaled": _stats(live_acc / 30.0),
        "oursAccelerationScaled": _stats(ours_acc / 30.0),
        "liveOrientation": _stats(live_ori),
        "oursOrientation": _stats(ours_ori),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("ours", type=Path)
    parser.add_argument("--sequence", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare(args.recording, args.ours, args.sequence)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
