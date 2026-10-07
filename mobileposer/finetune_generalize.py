"""Small-data generalization fine-tune for the realtime MobilePose export.

This deliberately uses action-level holdout validation, input-domain noise and
an anchor penalty to the starting checkpoint. It never overwrites an existing
checkpoint.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from mobileposer.mobile_export import MobilePose, windows
from mobileposer.no_head_layouts import LAYOUTS


def _capture_sequences(root: Path, layout: str):
    catalog = json.loads((root / "catalog.json").read_text(encoding="utf-8"))
    out = []
    for key, item in catalog.items():
        report_path = Path(item.get("report", ""))
        if not report_path.exists():
            normalized = str(report_path).replace("\\", "/")
            marker = "results/ours_multi_finetune/"
            if marker in normalized:
                report_path = root / normalized.split(marker, 1)[1]
        if not report_path.exists():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        entry = report.get("layouts", {}).get(layout)
        if not entry or entry.get("status") != "prepared":
            continue
        input_path = Path(entry["input"])
        target_path = Path(item["target"])
        marker = "results/ours_multi_finetune/"
        for path_value in (input_path, target_path):
            if path_value.exists():
                continue
            normalized = str(path_value).replace("\\", "/")
            if marker in normalized:
                resolved = root / normalized.split(marker, 1)[1]
                if path_value == input_path:
                    input_path = resolved
                else:
                    target_path = resolved
        if not input_path.exists() or not target_path.exists():
            continue
        data = torch.load(input_path, map_location="cpu", weights_only=False)
        target = torch.load(target_path, map_location="cpu", weights_only=False)
        imu = torch.cat(((data["acc"][0] / 30).flatten(1), data["ori"][0].flatten(1)), dim=-1)
        out.append((key, imu.float(), target["pose"].float()))
    return out


def _is_validation(key: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, key) for pattern in patterns)


def _predict_sequence(model, imu: torch.Tensor, max_frames: int):
    count = min(len(imu), max_frames) if max_frames > 0 else len(imu)
    result = []
    for window in windows(imu[:count].numpy()):
        result.append(model(torch.from_numpy(window)))
    return torch.stack(result)


def _loss(pred, target):
    # Frobenius loss on local rotation matrices; targets already match the
    # MobilePose decoder's fixed-joint convention.
    return (pred - target[:len(pred)]).square().mean()


@torch.no_grad()
def evaluate(model, sequences, max_frames):
    model.eval()
    values = []
    for _, imu, target in sequences:
        pred = _predict_sequence(model, imu, max_frames)
        values.append(float(torch.sqrt((pred - target[:len(pred)]).square().mean()).item()))
    return float(np.mean(values)) if values else float("inf")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, default=Path("results/ours_multi_finetune"))
    p.add_argument("--init", type=Path, default=Path("results/ours_multi_finetune/layouts/wrists_shanks_waist/model_finetuned.pth"))
    p.add_argument("--layout", choices=list(LAYOUTS), default="wrists_shanks_waist")
    p.add_argument("--output", type=Path, default=Path("results/generalize_finetune_v1"))
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--anchor-weight", type=float, default=0.01)
    p.add_argument("--acc-noise", type=float, default=0.01)
    p.add_argument("--ori-noise", type=float, default=0.003)
    p.add_argument("--max-frames", type=int, default=300)
    p.add_argument("--validation-pattern", action="append", default=[r"7_", r"9_", "卧位猫"])
    args = p.parse_args()
    torch.set_num_threads(1)
    all_sequences = _capture_sequences(args.dataset_root, args.layout)
    validation = [item for item in all_sequences if _is_validation(item[0], args.validation_pattern)]
    training = [item for item in all_sequences if item not in validation]
    if not training or not validation:
        raise ValueError(f"need non-empty train/validation split, got {len(training)}/{len(validation)}")
    model = MobilePose().load_combined(args.init).train()
    initial = {name: value.detach().clone() for name, value in model.state_dict().items() if name.startswith(("joints.", "pose."))}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best, best_epoch = float("inf"), 0
    history = []
    for epoch in range(args.epochs + 1):
        if epoch:
            model.train()
            losses = []
            for _, imu, target in training:
                noisy = imu.clone()
                noisy[:, :15] += torch.randn_like(noisy[:, :15]) * args.acc_noise
                noisy[:, 15:] += torch.randn_like(noisy[:, 15:]) * args.ori_noise
                optimizer.zero_grad(set_to_none=True)
                pred = _predict_sequence(model, noisy, args.max_frames)
                loss = _loss(pred, target)
                anchor = sum((model.state_dict()[name] - initial[name]).square().mean() for name in initial)
                total = loss + args.anchor_weight * anchor
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(total.detach()))
        train_error = evaluate(model, training, args.max_frames)
        val_error = evaluate(model, validation, args.max_frames)
        history.append({"epoch": epoch, "trainRmse": train_error, "validationRmse": val_error})
        print(f"epoch={epoch}/{args.epochs} train={train_error:.6f} validation={val_error:.6f}", flush=True)
        if val_error < best:
            best, best_epoch = val_error, epoch
            args.output.mkdir(parents=True, exist_ok=True)
            state = {}
            for name, value in model.state_dict().items():
                if name.startswith("joints."):
                    state["joints.joints." + name[len("joints."):]] = value.detach().cpu()
                elif name.startswith("pose."):
                    state["pose.pose." + name[len("pose."):]] = value.detach().cpu()
            torch.save(state, args.output / "model_finetuned.pth")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "history.json").write_text(json.dumps({"args": vars(args) | {"dataset_root": str(args.dataset_root), "init": str(args.init), "output": str(args.output)}, "bestEpoch": best_epoch, "history": history}, indent=2, default=str), encoding="utf-8")
    print(f"saved={args.output / 'model_finetuned.pth'} best_epoch={best_epoch} validation={best:.6f}")


if __name__ == "__main__":
    main()
