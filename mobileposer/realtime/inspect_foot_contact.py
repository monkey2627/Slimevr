"""Inspect the trained foot-contact head on a recorded raw-IMU capture."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from mobileposer.realtime.calibration import calibrate_heading
from mobileposer.realtime.infer import InferenceSession, build_feature, load_model
from mobileposer.mobile_export import MobilePose
from mobileposer.realtime.layout import ResolvedLayout
from mobileposer.realtime.replay_raw_imu import load_capture


class FootContact(nn.Module):
    """Lightning-free equivalent of models.footcontact.FootContact."""

    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(132, 64)
        self.rnn = nn.LSTM(64, 64, num_layers=2, bidirectional=True)
        self.linear2 = nn.Linear(128, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.linear1(x)).transpose(0, 1)
        x, _ = self.rnn(x)
        return self.linear2(x.transpose(0, 1))


def load_foot_contact(path: Path) -> FootContact:
    state = torch.load(path, map_location="cpu", weights_only=True)
    prefix = "foot_contact.footcontact."
    selected = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    if not selected:
        raise ValueError(f"{path} has no {prefix} weights")
    model = FootContact()
    model.load_state_dict(selected, strict=True)
    return model.eval()


def main() -> None:
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("--layout", default="wrists_shanks_waist")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("checkpoints/no_head_5imu_surface/wrists_shanks_waist/1/base_model.pth"))
    parser.add_argument("--max-frames", type=int, default=600)
    args = parser.parse_args()

    labels, raw, adjusted, accel, times, sensor_ids, _ = load_capture(args.capture, args.layout)
    limit = min(args.max_frames, len(times))
    raw, adjusted, accel, times = raw[:limit], adjusted[:limit], accel[:limit], times[:limit]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    foot = load_foot_contact(args.checkpoint)
    model = MobilePose().load_combined(args.checkpoint)
    calibration = calibrate_heading(labels,
                                     {label: raw[:, i] for i, label in enumerate(labels)},
                                     {label: adjusted[:, i] for i, label in enumerate(labels)})
    resolved = ResolvedLayout(args.layout, labels,
                              {label: sensor_ids[i] for i, label in enumerate(labels)})
    session = InferenceSession(resolved, model, calibration)
    probabilities, velocities = [], []
    with torch.no_grad():
        for frame in range(len(times)):
            session.step(adjusted[frame], accel[frame])
            window = torch.from_numpy(session.window._window[None])
            joints = model.joints(window)
            logits = foot(torch.cat((joints, window), dim=-1))[0, -1]
            probabilities.append(torch.sigmoid(logits).numpy())
            velocities.append(float(torch.linalg.vector_norm(joints[0, -1, 30:36] - joints[0, -2, 30:36])))
    p = np.asarray(probabilities)
    print(f"checkpoint={args.checkpoint}")
    print(f"frames={len(p)}, duration={times[-1]-times[0]:.2f}s, labels={labels}")
    print(f"strict_load=ok, foot_contact_keys={sum(k.startswith('foot_contact.footcontact.') for k in checkpoint)}")
    for index, name in enumerate(("left", "right")):
        print(f"{name}: min={p[:, index].min():.4f}, max={p[:, index].max():.4f}, "
              f"mean={p[:, index].mean():.4f}, over_0.5={(p[:, index] > .5).mean():.3f}")
    print(f"contact_correlation_with_predicted_foot_motion={np.corrcoef(p.mean(axis=1), velocities)[0,1]:.4f}")


if __name__ == "__main__":
    main()
