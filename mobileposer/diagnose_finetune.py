"""Compare base and finetuned checkpoints per capture and SMPL joint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mobileposer.articulate import math
from mobileposer.config import joint_set, model_config
from mobileposer.mobile_export import MobilePose, windows
from mobileposer.no_head_layouts import LAYOUTS

JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand",
]


def _load_model(path: Path, device: torch.device):
    return MobilePose().load_combined(path).to(device).eval()


@torch.no_grad()
def _predict(model, imu, target, device):
    predictions = []
    for window in windows(imu[0].cpu().numpy()):
        predictions.append(model(torch.from_numpy(window).to(device)).cpu())
    pose = torch.stack(predictions)
    reduced_target = target["pose"][:, joint_set.reduced].to(device)
    reduced_pred = pose[:, joint_set.reduced].to(device)
    delta = reduced_pred.transpose(-1, -2) @ reduced_target
    angle = torch.acos(((delta.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0))
    rotation_error_deg = angle.mean(dim=0) * 180.0 / np.pi
    return {
        "jointErrorCm": {},
        "rotationErrorDeg": {
            JOINT_NAMES[joint]: float(rotation_error_deg[index].cpu())
            for index, joint in enumerate(joint_set.reduced)
        },
        "mpjpeCm": float("nan"),
        "rotationMeanDeg": float(rotation_error_deg.mean().cpu()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("results/ours_multi_finetune"))
    parser.add_argument("--base", type=Path, default=Path("checkpoints/no_head_5imu_surface/wrists_shanks_waist/1/base_model.pth"))
    parser.add_argument("--finetuned", type=Path, default=Path("results/ours_multi_finetune/layouts/wrists_shanks_waist/model_finetuned.pth"))
    parser.add_argument("--label", default="candidate", help="name for the --finetuned checkpoint in output")
    parser.add_argument("--layout", choices=list(LAYOUTS), default="wrists_shanks_waist")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("results/finetune_diagnostics.json"))
    args = parser.parse_args()
    device = torch.device(args.device)
    catalog_path = args.dataset_root / "catalog.json"
    if not catalog_path.exists():
        raise FileNotFoundError(catalog_path)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    # Catalogs generated on the original Linux training machine contain
    # absolute Linux paths; resolve them against this checkout on Windows.
    for item in catalog.values():
        for field in ("report", "target"):
            value = item.get(field)
            if value and not Path(value).exists():
                marker = "results/ours_multi_finetune/"
                normalized = str(value).replace("\\", "/")
                if marker in normalized:
                    item[field] = str(args.dataset_root / normalized.split(marker, 1)[1])
        report_path = Path(item.get("report", ""))
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            for entry in report.get("layouts", {}).values():
                value = entry.get("input")
                if value and not Path(value).exists():
                    normalized = str(value).replace("\\", "/")
                    if marker in normalized:
                        entry["input"] = str(args.dataset_root / normalized.split(marker, 1)[1])
            item["_resolvedReport"] = report

    sequences = []
    for key, item in catalog.items():
        report = item.get("_resolvedReport")
        if not report or args.layout not in item.get("layouts", {}) and args.layout not in item.get("layouts", []):
            continue
        entry = report.get("layouts", {}).get(args.layout)
        if not entry or entry.get("status") != "prepared":
            continue
        input_path = Path(entry["input"])
        data = torch.load(input_path, map_location="cpu", weights_only=False)
        target_path = Path(item["target"])
        target = torch.load(target_path, map_location="cpu", weights_only=False)
        imu = torch.cat(((data["acc"][0] / 30).flatten(1), data["ori"][0].flatten(1)), -1)[None]
        sequences.append((key, imu, target))
    if not sequences:
        raise ValueError(f"No prepared sequences for {args.layout}")
    base = _load_model(args.base, device)
    finetuned = _load_model(args.finetuned, device)
    rows = []
    for key, imu, target in sequences:
        before = _predict(base, imu, target, device)
        after = _predict(finetuned, imu, target, device)
        rows.append({"capture": key, "before": before, "after": after})
        print(f"{key}: base rotation={before['rotationMeanDeg']:.3f}deg fine={after['rotationMeanDeg']:.3f}deg", flush=True)
    weights = np.asarray([len(imu) for _, imu, _ in sequences], dtype=np.float64)
    def aggregate(stage):
        values = [row[stage] for row in rows]
        result = {
            "rotationMeanDeg": float(np.average([x["rotationMeanDeg"] for x in values], weights=weights)),
            "rotationErrorDeg": {},
        }
        for name in values[0]["rotationErrorDeg"]:
            result["rotationErrorDeg"][name] = float(np.average([x["rotationErrorDeg"][name] for x in values], weights=weights))
        return result
    output = {"layout": args.layout, "base": str(args.base), "finetuned": str(args.finetuned), "finetunedLabel": args.label,
              "captures": rows, "aggregate": {"before": aggregate("before"), "after": aggregate("after")}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(output["aggregate"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
