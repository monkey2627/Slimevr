"""End-to-end realtime bridge: SlimeVR-Server (SolarXR) -> mobileposer
inference -> local WebSocket broadcast for the Unity receiver script.

Does not modify SlimeVR-Server -- it only subscribes to its existing SolarXR
data feed (see solarxr_client.py) as an ordinary client. SlimeVR-Server's own
VMC/OSC output keeps running unmodified in parallel (useful as a comparison
reference, plan M6).

Orientation needs no calibration (rotation_reference_adjusted is already
usable directly, see calibration.py::bone_orientation). Acceleration DOES
need a short startup calibration window with rotational motion -- this is not
avoidable, see calibration.py's module docstring for why (SlimeVR-Server
composes the raw<->reference-adjusted relationship from four private,
unexposed quaternions; only the delta-trajectory trick used here can recover
the one recoverable component). Once fitted, the ongoing per-frame cost is a
single fixed matrix multiply -- streaming is real-time after that.

Usage (from repo root, base_mobileposer/), with SlimeVR-Server running,
trackers bound, and the matching no_head_5imu_surface checkpoint synced
locally to checkpoints/no_head_5imu_surface/<layout>/1/base_model.pth::

    python -m mobileposer.realtime.run

NOTE: this is the M1-M4 version per the implementation plan. It resolves the
layout once at startup and does not yet hot-swap if trackers connect/disconnect
mid-session (plan M5) -- restart the script if your tracker set changes.
"""
from __future__ import annotations

import argparse
import asyncio
import atexit
import dataclasses
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import websockets
import yaml

from mobileposer.realtime.calibration import UnobservableHeadingError, calibrate_heading
from mobileposer.realtime.infer import DEFAULT_CHECKPOINT_ROOT, OUTPUT_LAG_FRAMES, InferenceSession, load_model
from mobileposer.realtime.layout import LayoutMismatchError, ResolvedLayout, detect_layout
from mobileposer.realtime.rotmat import batch_smpl_matrix_to_unity_quat_xyzw, quat_xyzw_to_unity_quat_xyzw
from mobileposer.realtime.solarxr_client import TrackerSample, stream_trackers
from mobileposer.realtime.stream_out import PoseBroadcaster
from mobileposer.realtime.web_visualizer import start_web_visualizer
from mobileposer.realtime.recording import RealtimeRecorder
from mobileposer.mobile_export import MobilePose
from mobileposer.realtime.inspect_foot_contact import load_foot_contact


def normalize_checkpoint_path(value: str | Path) -> Path:
    """Normalize ASCII and typographic shell quotes without source encoding issues."""
    quote_chars = "\"'" + "".join(chr(code) for code in (0x2018, 0x2019, 0x201C, 0x201D))
    text = str(value).strip()
    while text and text[0] in quote_chars:
        text = text[1:]
    while text and text[-1] in quote_chars:
        text = text[:-1]
    return Path(text)


def _predict_contact(contact_model, joints: torch.Tensor, window: torch.Tensor) -> np.ndarray:
    """Run the optional contact head away from the asyncio receive loop."""
    with torch.inference_mode():
        logits = contact_model(torch.cat((joints, window), dim=-1))[0, -1]
        return torch.sigmoid(logits).numpy()


def _load_runtime_config(path: Path | None) -> dict:
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as file:
        values = yaml.safe_load(file) or {}
    if not isinstance(values, dict):
        raise ValueError(f"realtime config must be a mapping: {path}")
    return values


async def _resilient_tracker_updates(recorder_holder=None):
    while True:
        try:
            received = False
            async for update in stream_trackers():
                received = True
                yield update
            if recorder_holder and recorder_holder.get("latest") is not None:
                recorder_holder["latest"].clear()
            if received:
                print("SolarXR stream ended; reconnecting in 1s...", file=sys.stderr)
                if recorder_holder and recorder_holder.get("recorder"):
                    recorder_holder["recorder"].event("solarxr_stream_ended")
        except (websockets.ConnectionClosed, OSError, ConnectionError) as exc:
            print(
                f"SolarXR disconnected ({type(exc).__name__}: {exc}); reconnecting in 1s...",
                file=sys.stderr,
            )
            if recorder_holder and recorder_holder.get("latest") is not None:
                recorder_holder["latest"].clear()
            if recorder_holder and recorder_holder.get("recorder"):
                recorder_holder["recorder"].event("solarxr_disconnected", error=repr(exc))
        await asyncio.sleep(1.0)


class _TrackerUpdatePump:
    """Continuously drain SolarXR so inference cannot starve its socket."""

    def __init__(self, source, capacity: int = 8):
        self._source = source
        self._queue = asyncio.Queue(maxsize=capacity)
        self._task = asyncio.create_task(self._run())

    async def _run(self):
        async for update in self._source:
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            await self._queue.put(update)

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self._queue.get()


async def _wait_for_layout(tracker_stream, latest: Dict[str, TrackerSample]) -> ResolvedLayout:
    print("Waiting for a recognized tracker layout (bind trackers in SlimeVR-Server)...", file=sys.stderr)
    last_error = None
    async for update in tracker_stream:
        latest.update(update)
        try:
            resolved = detect_layout(latest)
        except LayoutMismatchError as exc:
            if str(exc) != last_error:
                print(f"  ... {exc}", file=sys.stderr)
                last_error = str(exc)
            continue
        print(f"Resolved layout: {resolved.name} (labels: {resolved.labels})", file=sys.stderr)
        return resolved
    raise RuntimeError("SolarXR stream ended before a layout was resolved")


async def _collect_heading_window(
    tracker_stream, latest: Dict[str, TrackerSample], resolved: ResolvedLayout, frames: int,
    recorder: RealtimeRecorder | None = None,
):
    """Collect `frames` samples per label while the user moves the trackers
    around (any rotation works -- twisting the torso, swinging limbs, etc.).
    A perfectly still hold will not work: see calibration.py."""
    raw: Dict[str, List] = {label: [] for label in resolved.labels}
    adjusted: Dict[str, List] = {label: [] for label in resolved.labels}
    print(
        f"Calibrating: rotate/move the trackers for the next {frames} frames "
        f"(twist your torso, swing your arms/legs -- holding still will NOT work)...",
        file=sys.stderr,
    )
    async for update in tracker_stream:
        latest.update(update)
        for label in resolved.labels:
            key = resolved.label_to_tracker_key[label]
            sample = latest.get(key)
            if sample is not None and sample.online:
                raw[label].append(sample.raw_quat_xyzw)
                adjusted[label].append(sample.quat_xyzw)
        if all(len(v) >= frames for v in raw.values()):
            break
    raw_arrays = {label: np.asarray(values[:frames], dtype=np.float32) for label, values in raw.items()}
    adjusted_arrays = {label: np.asarray(values[:frames], dtype=np.float32) for label, values in adjusted.items()}
    return raw_arrays, adjusted_arrays


async def _wait_for_sync_calibration(tracker_stream, latest, resolved, recorder=None) -> None:
    """Coordinate SlimeVR's manual upright/ski calibration with this bridge.

    SolarXR does not expose calibration-complete events, so the user confirms
    each SlimeVR step explicitly. The captured windows are diagnostic/reference
    samples; the existing heading fit still follows and remains unchanged.
    """
    print("\nSynchronized calibration mode", file=sys.stderr)
    print("1) In SlimeVR, perform the upright/standing calibration.", file=sys.stderr)
    await asyncio.to_thread(input, "   Press Enter here when it is complete... ")
    upright = await _collect_reference_window(tracker_stream, latest, resolved, 15, recorder)
    print("   Upright reference captured.", file=sys.stderr)
    print("2) In SlimeVR, perform the ski calibration.", file=sys.stderr)
    await asyncio.to_thread(input, "   Press Enter here when it is complete... ")
    ski = await _collect_reference_window(tracker_stream, latest, resolved, 15, recorder)
    # Report only relative motion; absolute quaternion signs are arbitrary.
    labels = resolved.labels
    deltas = np.linalg.norm(upright - ski, axis=2).mean(axis=0)
    print("   Ski reference captured. Mean quaternion delta by tracker: "
          + ", ".join(f"{label}={value:.3f}" for label, value in zip(labels, deltas)),
          file=sys.stderr)


async def _collect_reference_window(tracker_stream, latest, resolved, frames, recorder=None):
    values = []
    async for update in tracker_stream:
        latest.update(update)
        if recorder is not None:
            selected = [latest.get(resolved.label_to_tracker_key[label]) for label in resolved.labels]
            recorder.write("calibrationSample", {
                "hostUnixSeconds": time.time(),
                "hostMonotonicSeconds": time.perf_counter(),
                "trackers": [dataclasses.asdict(sample) for sample in selected if sample is not None],
            })
        row = []
        complete = True
        for label in resolved.labels:
            sample = latest.get(resolved.label_to_tracker_key[label])
            if sample is None or not sample.online:
                complete = False
                break
            row.append(sample.quat_xyzw)
        if complete:
            values.append(np.asarray(row, dtype=np.float32))
        if len(values) >= frames:
            return np.asarray(values[-frames:], dtype=np.float32)
    raise RuntimeError("SolarXR stream ended during synchronized calibration")


async def _collect_upright_calibration(tracker_stream, latest, resolved, frames, settle_seconds):
    print("\nStand upright in the neutral pose (feet parallel, arms relaxed).", file=sys.stderr)
    settle_seconds = max(0.0, float(settle_seconds))
    if settle_seconds > 0:
        for remaining in range(int(np.ceil(settle_seconds)), 0, -1):
            print(f"  Capturing upright reference in {remaining}...", file=sys.stderr)
            await asyncio.sleep(1.0)
    print("  Capturing upright reference now...", file=sys.stderr)
    rows = []
    async for update in tracker_stream:
        latest.update(update)
        samples = [latest.get(resolved.label_to_tracker_key[label]) for label in resolved.labels]
        if any(sample is None or not sample.online for sample in samples):
            continue
        rows.append((
            np.asarray([sample.quat_xyzw for sample in samples], dtype=np.float32),
            np.asarray([sample.accel_xyz for sample in samples], dtype=np.float32),
        ))
        if len(rows) >= frames:
            print("Upright reference captured.", file=sys.stderr)
            return rows[-1]
    raise RuntimeError("SolarXR stream ended during upright calibration")


async def main_async(args: argparse.Namespace) -> None:
    broadcaster = PoseBroadcaster()
    await broadcaster.serve(args.host, args.port)
    print(f"Broadcasting SMPL pose on ws://{args.host}:{args.port}", file=sys.stderr)
    if args.web:
        start_web_visualizer(args.web_host, args.web_port, args.port)
        print(f"Realtime viewer: http://{args.web_host}:{args.web_port}/", file=sys.stderr)

    latest: Dict[str, TrackerSample] = {}
    recorder_holder = {"recorder": None, "latest": None}
    tracker_stream = _TrackerUpdatePump(
        _resilient_tracker_updates(recorder_holder)
    )
    recorder_holder["latest"] = latest

    resolved = await _wait_for_layout(tracker_stream, latest)

    checkpoint = normalize_checkpoint_path(args.checkpoint) if args.checkpoint is not None else None
    contact_model = None
    if args.include_contact:
        contact_sources = []
        if checkpoint is not None:
            contact_sources.append(checkpoint)
        contact_sources.append(
            Path("results/ours_multi_finetune/layouts")
            / resolved.name / "model_finetuned.pth"
        )
        contact_sources.append(args.checkpoint_root / resolved.name / "1" / "base_model.pth")
        for contact_source in contact_sources:
            try:
                contact_model = load_foot_contact(contact_source)
                print(f"Foot contact head loaded from: {contact_source}", file=sys.stderr)
                break
            except (ValueError, FileNotFoundError) as exc:
                print(f"Foot contact head unavailable in {contact_source}: {exc}", file=sys.stderr)
        if contact_model is None:
            print("Warning: no foot-contact head found; continuing without footContact data.",
                  file=sys.stderr)
    recorder = None
    if args.record is not None:
        recorder = RealtimeRecorder(args.record, {
            "layout": resolved.name,
            "labels": resolved.labels,
            "labelToTrackerKey": resolved.label_to_tracker_key,
            "checkpoint": checkpoint,
            "checkpointRoot": args.checkpoint_root,
            "outputFps": args.output_fps,
            "calibrationFrames": args.calibration_frames,
            "outputLagFrames": OUTPUT_LAG_FRAMES,
        })
        recorder_holder["recorder"] = recorder
        atexit.register(recorder.close)
        recorder.event("layout_resolved")

    upright_reference = None
    if not args.skip_upright_calibration:
        upright_reference = await _collect_upright_calibration(
            tracker_stream, latest, resolved, args.upright_calibration_frames,
            args.upright_settle_seconds,
        )
        if recorder is not None:
            recorder.write("upright", {
                "referenceAdjustedQuaternions": upright_reference[0],
                "linearAccelerationMps2": upright_reference[1],
            })

    if args.sync_calibration:
        await _wait_for_sync_calibration(tracker_stream, latest, resolved, recorder)

    while True:
        raw_arrays, adjusted_arrays = await _collect_heading_window(
            tracker_stream, latest, resolved, args.calibration_frames, recorder
        )
        try:
            heading = calibrate_heading(resolved.labels, raw_arrays, adjusted_arrays)
            break
        except UnobservableHeadingError as exc:
            print(f"  Calibration failed, retrying: {exc}", file=sys.stderr)
    for label, diag in heading.diagnostics.items():
        print(f"  {label}: yaw={diag['yaw_deg']:+.1f}deg residual_p95={diag['delta_residual_p95_deg']:.2f}deg",
              file=sys.stderr)
    if recorder is not None:
        recorder.write("heading", {
            "labels": heading.labels,
            "matrices": heading.heading,
            "diagnostics": heading.diagnostics,
        })

    # Windows shells pass typographic quotes literally when a command is
    # copied from formatted text. Strip them so the resulting Path refers to
    # the actual checkpoint file rather than a filename containing `“`/`”`.
    model = (MobilePose().load_combined(checkpoint) if checkpoint is not None
             else load_model(resolved.name, args.checkpoint_root))
    session = InferenceSession(layout=resolved, model=model, heading=heading)
    if upright_reference is not None:
        session.prime(*upright_reference)
    print("Calibrated. Streaming live pose...", file=sys.stderr)

    frame_index = 0
    source_update_index = -1
    last_contact = None
    contact_task = None
    next_output_time = time.perf_counter()
    async for update in tracker_stream:
        update_received_monotonic = time.perf_counter()
        latest.update(update)
        source_update_index += 1
        if recorder is not None:
            recorder.write("solarxrUpdate", {
                "sourceUpdate": source_update_index,
                "receiveMonotonicSeconds": update_received_monotonic,
                "receiveUnixSeconds": time.time(),
                "trackers": [dataclasses.asdict(sample) for sample in update.values()],
            })
        now = time.perf_counter()
        if now < next_output_time:
            continue
        next_output_time = now + (1.0 / max(1.0, args.output_fps))
        try:
            quats = np.stack(
                [latest[resolved.label_to_tracker_key[label]].quat_xyzw for label in resolved.labels]
            )
            accel = np.stack(
                [latest[resolved.label_to_tracker_key[label]].accel_xyz for label in resolved.labels]
            )
        except KeyError:
            continue  # a required tracker dropped out; wait for it to come back

        inference_start_monotonic = time.perf_counter()
        host_unix = time.time()
        feature, calibrated_ori, aligned_accel = session.prepare_feature(quats, accel)
        # MobilePose is CPU-bound and can take long enough to starve the
        # asyncio event loop.  Keep SolarXR's WebSocket reader responsive by
        # running the blocking model call in a worker thread.
        output = await asyncio.to_thread(session.step_feature, feature)
        inference_end_monotonic = time.perf_counter()
        quats_out = batch_smpl_matrix_to_unity_quat_xyzw(output.detach().numpy())
        tracker_rotations = np.stack([quat_xyzw_to_unity_quat_xyzw(q) for q in quats])
        if contact_task is not None and contact_task.done():
            try:
                last_contact = contact_task.result()
            finally:
                contact_task = None
        if (contact_model is not None
                and contact_task is None
                and frame_index % max(1, args.contact_interval) == 0):
            window = torch.from_numpy(session.window._window[None].copy())
            joints = getattr(session.model, "_last_joints", None)
            if joints is None or joints.shape[:2] != window.shape[:2]:
                joints = session.model.joints(window)
            contact_task = asyncio.create_task(
                asyncio.to_thread(_predict_contact, contact_model, joints, window))
        contact = last_contact if contact_model is not None else None
        diagnostics = {
            "sourceUpdate": source_update_index,
            "modelInputFrame": frame_index,
            "modelOutputRepresentsFrame": (
                frame_index - OUTPUT_LAG_FRAMES
                if frame_index >= OUTPUT_LAG_FRAMES else -1
            ),
            "receiveMonotonicSeconds": update_received_monotonic,
            "receiveUnixSeconds": host_unix,
            "inferenceStartMonotonicSeconds": inference_start_monotonic,
            "inferenceEndMonotonicSeconds": inference_end_monotonic,
            "modelLagFrames": OUTPUT_LAG_FRAMES,
            "footContactEnabled": contact_model is not None,
        }
        if recorder is not None:
            tracker_rows = []
            for label in resolved.labels:
                sample = latest[resolved.label_to_tracker_key[label]]
                tracker_rows.append({
                    "label": label,
                    "trackerKey": sample.tracker_key,
                    "deviceId": sample.device_id,
                    "trackerNum": sample.tracker_num,
                    "bodyPart": sample.body_part,
                    "status": sample.status,
                    "online": sample.online,
                    "rawQuaternion": sample.raw_quat_xyzw,
                    "referenceAdjustedQuaternion": sample.quat_xyzw,
                    "linearAccelerationMps2": sample.accel_xyz,
                })
            local = output.detach().numpy()
            global_rot = np.empty_like(local)
            parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
            for joint, parent in enumerate(parents):
                global_rot[joint] = local[joint] if parent < 0 else global_rot[parent] @ local[joint]
            frame_record = {
                **diagnostics,
                "frame": frame_index,
                "trackers": tracker_rows,
                "alignedAccelerationMps2": aligned_accel,
                "calibratedOrientationMatrices": calibrated_ori,
                "feature": feature,
                "smplLocalRotationMatrices": local,
                "smplGlobalRotationMatrices": global_rot,
                "unityLocalQuaternions": quats_out,
                "footContact": contact,
            }
        await broadcaster.broadcast(
            frame_index, resolved.name, quats_out, contact, tracker_rotations, diagnostics)
        if recorder is not None:
            frame_record["broadcastMonotonicSeconds"] = time.perf_counter()
            recorder.write("inferenceFrame", frame_record)
        frame_index += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1", help="host to broadcast the SMPL pose stream on")
    parser.add_argument("--port", type=int, default=21200, help="port to broadcast the SMPL pose stream on")
    parser.add_argument("--web", action="store_true", help="serve a browser-based realtime stick-figure viewer")
    parser.add_argument("--web-host", default="127.0.0.1", help="web viewer bind address")
    parser.add_argument("--web-port", type=int, default=8765, help="web viewer HTTP port")
    parser.add_argument("--calibration-frames", type=int, default=150,
                         help="frames to collect (while moving/rotating) for the startup heading fit, ~5s at 30Hz")
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT,
                         help="root directory containing <layout>/1/base_model.pth checkpoints")
    parser.add_argument("--checkpoint", type=Path, default=None,
                         help="explicit combined checkpoint for the resolved layout")
    parser.add_argument("--include-contact", action="store_true",
                         help="broadcast footContact=[left,right] probabilities")
    parser.add_argument("--contact-interval", type=int, default=30,
                         help="recompute contact every N live frames (default: 30)")
    parser.add_argument("--output-fps", type=float, default=30.0,
                         help="maximum pose broadcasts/inferences per second (default: 30)")
    parser.add_argument("--sync-calibration", action="store_true",
                         help="pause for manual SlimeVR upright and ski calibration before heading fit")
    parser.add_argument("--upright-calibration-frames", type=int, default=30,
                         help="neutral-pose samples used to prime the model context (default: 30)")
    parser.add_argument("--upright-settle-seconds", type=float, default=3.0,
                         help="automatic countdown before upright capture (default: 3 seconds)")
    parser.add_argument("--skip-upright-calibration", action="store_true",
                         help="skip the neutral-pose prompt and use the legacy startup behavior")
    parser.add_argument("--record", type=Path, default=None,
                         help="write live SolarXR inputs, exact model features and SMPL outputs as JSONL")
    parser.add_argument("--config", type=Path, default=None,
                         help="YAML defaults for the realtime bridge; explicit CLI values take precedence")
    args = parser.parse_args()
    config = _load_runtime_config(args.config)
    defaults = {
        "calibration_frames": 150,
        "checkpoint_root": DEFAULT_CHECKPOINT_ROOT,
        "output_fps": 30.0,
        "upright_calibration_frames": 30,
        "upright_settle_seconds": 3.0,
        "port": 21200,
    }
    for name, default in defaults.items():
        if getattr(args, name) == default and name in config:
            value = config[name]
            if name in {"checkpoint_root"}:
                value = Path(value)
            setattr(args, name, value)
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
