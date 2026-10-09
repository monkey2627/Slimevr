"""Runtime MobilePose network used by the realtime Python bridge."""
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
PARENTS = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
REDUCED = [0, 1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
IGNORED = [7, 8, 10, 11, 20, 21, 22, 23]
WINDOW, OUTPUT_INDEX = 45, 40


class DenseRnn(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.linear1 = nn.Linear(inputs, 256)
        self.rnn = nn.LSTM(256, 256, num_layers=2, bidirectional=True)
        self.linear2 = nn.Linear(512, outputs)

    def forward(self, x):
        x = torch.relu(self.linear1(x)).transpose(0, 1)
        x, _ = self.rnn(x)
        return self.linear2(x.transpose(0, 1))


class MobilePose(nn.Module):
    """Inference-only wrapper for the combined 60D-to-SMPL checkpoint."""

    def __init__(self):
        super().__init__()
        self.joints = DenseRnn(60, 72)
        self.pose = DenseRnn(132, 96)
        self.register_buffer('identity', torch.eye(3))

    def load_combined(self, path):
        state = torch.load(path, map_location='cpu', weights_only=True)
        for module, prefix in [(self.joints, 'joints.joints.'), (self.pose, 'pose.pose.')]:
            module.load_state_dict(
                {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)},
                strict=True,
            )
        return self.eval()

    def forward(self, imu):
        joints = self.joints(imu)
        self._last_joints = joints
        six = self.pose(torch.cat([joints, imu], dim=-1))[0, OUTPUT_INDEX].reshape(16, 6)
        c0 = six[:, :3] / torch.linalg.vector_norm(six[:, :3], dim=-1, keepdim=True)
        c1 = six[:, 3:] - (c0 * six[:, 3:]).sum(-1, keepdim=True) * c0
        c1 = c1 / torch.linalg.vector_norm(c1, dim=-1, keepdim=True)
        rot = torch.stack([c0, c1, torch.cross(c0, c1, dim=-1)], dim=-1)
        rot = torch.where(torch.isnan(rot), torch.zeros_like(rot), rot)
        global_rot = [rot[REDUCED.index(j)] if j in REDUCED else self.identity for j in range(24)]
        local = [global_rot[0]]
        for j in range(1, 24):
            local.append(self.identity if j in IGNORED else global_rot[PARENTS[j]].T @ global_rot[j])
        return torch.stack(local)


def windows(frames):
    window = None
    for frame in frames:
        window = np.repeat(frame[None], WINDOW, axis=0) if window is None else np.concatenate([window[1:], frame[None]])
        yield window[None].astype(np.float32)
