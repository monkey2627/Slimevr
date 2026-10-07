"""Replay a stored SMPL pose sequence to the Unity realtime receiver.

This intentionally bypasses SolarXR, SlimeVR, calibration, and the neural
network. It is a diagnostic for the final SMPL-local-pose -> Unity retargeting
stage. The input is a processed MobilePoser ``.pt`` file containing a ``pose``
entry (a list of ``[T, 24, 3, 3]`` local rotation tensors).

Example::

    python -m mobileposer.realtime.replay_pose data/.../ACCAD.pt --sequence 0

The receiver must be running in Unity with MobilePoserPoseSource configured for
ws://127.0.0.1:21200. No SlimeVR server is needed for this test.
"""
from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

import numpy as np
import torch
from mobileposer.realtime.rotmat import batch_smpl_matrix_to_unity_quat_xyzw
from mobileposer.realtime.stream_out import PoseBroadcaster


async def replay(path: Path, sequence: int, host: str, port: int, fps: float, loop: bool) -> None:
    data = torch.load(path, map_location="cpu", weights_only=False)
    poses = data.get("pose")
    if poses is None:
        raise KeyError(f"{path} has no 'pose' entry")
    if isinstance(poses, torch.Tensor):
        poses = [poses]
    if sequence < 0 or sequence >= len(poses):
        raise IndexError(f"sequence {sequence} outside 0..{len(poses) - 1}")

    pose = torch.as_tensor(poses[sequence]).float().numpy()
    if pose.ndim != 4 or pose.shape[1:] != (24, 3, 3):
        raise ValueError(f"expected pose[{sequence}] shape [T,24,3,3], got {pose.shape}")
    quats = np.stack([batch_smpl_matrix_to_unity_quat_xyzw(frame) for frame in pose])
    interval = 1.0 / fps

    broadcaster = PoseBroadcaster()
    server = await broadcaster.serve(host, port)
    print(f"Replay server listening on ws://{host}:{port}")
    print(f"Replaying {len(quats)} frames from {path} sequence {sequence} at {fps:g} Hz")
    try:
        while True:
            started = time.perf_counter()
            for frame_index, frame in enumerate(quats):
                await broadcaster.broadcast(frame_index, "offline-replay", frame)
                await asyncio.sleep(max(0.0, interval - (time.perf_counter() - started)))
                started = time.perf_counter()
            if not loop:
                break
    finally:
        server.close()
        await server.wait_closed()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="processed MobilePoser .pt file")
    parser.add_argument("--sequence", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=21200)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()
    asyncio.run(replay(args.data, args.sequence, args.host, args.port, args.fps, args.loop))


if __name__ == "__main__":
    main()
