"""Run a MobilePose checkpoint offline and replay predicted poses to Unity."""
from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

import numpy as np
import torch

from mobileposer.mobile_export import LAYOUTS, MobilePose, WINDOW
from mobileposer.realtime.rotmat import batch_smpl_matrix_to_unity_quat_xyzw
from mobileposer.realtime.stream_out import PoseBroadcaster


def load_features(path: Path, sequence: int) -> tuple[str, np.ndarray]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    layout = data.get("layout")
    if layout not in LAYOUTS:
        raise ValueError(f"unsupported or missing layout: {layout!r}")
    acc = torch.as_tensor(data["acc"][sequence]).float()[:, :5].numpy() / 30.0
    ori = torch.as_tensor(data["ori"][sequence]).float()[:, :5].numpy()
    if acc.shape[0] != ori.shape[0] or acc.shape[1:] != (5, 3) or ori.shape[1:] != (5, 3, 3):
        raise ValueError(f"invalid acc/ori shapes: {acc.shape}, {ori.shape}")
    features = np.concatenate([acc.reshape(acc.shape[0], 15), ori.reshape(ori.shape[0], 45)], axis=1)
    return layout, features.astype(np.float32)


def predict(model: MobilePose, features: np.ndarray) -> np.ndarray:
    window = None
    output = []
    with torch.no_grad():
        for frame in features:
            window = np.repeat(frame[None], WINDOW, axis=0) if window is None else np.concatenate([window[1:], frame[None]])
            output.append(model(torch.from_numpy(window[None])).numpy())
    return np.stack(output)


async def replay(args: argparse.Namespace) -> None:
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    sequences = data.get("acc")
    if sequences is None or args.sequence < 0 or args.sequence >= len(sequences):
        raise IndexError(f"sequence {args.sequence} outside available data")
    layout, features = load_features(args.data, args.sequence)
    checkpoint = args.checkpoint or Path("checkpoints/no_head_5imu_surface") / layout / "1" / "base_model.pth"
    model = MobilePose().load_combined(checkpoint)
    poses = predict(model, features)
    quats = np.stack([batch_smpl_matrix_to_unity_quat_xyzw(frame) for frame in poses])

    broadcaster = PoseBroadcaster()
    server = await broadcaster.serve(args.host, args.port)
    print(f"Replay inference server listening on ws://{args.host}:{args.port}")
    print(f"Layout={layout}, sequence={args.sequence}, frames={len(quats)}, checkpoint={checkpoint}")
    interval = 1.0 / args.fps
    try:
        while True:
            started = time.perf_counter()
            for frame_index, frame in enumerate(quats):
                await broadcaster.broadcast(frame_index, layout + "-offline-infer", frame)
                await asyncio.sleep(max(0.0, interval - (time.perf_counter() - started)))
                started = time.perf_counter()
            if not args.loop:
                break
    finally:
        server.close()
        await server.wait_closed()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--sequence", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=21200)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()
    asyncio.run(replay(args))


if __name__ == "__main__":
    main()
