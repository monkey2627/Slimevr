# Realtime diagnostics and A/B workflow

Record the exact SolarXR values, calibrated model features, SMPL output and
host timing without changing inference:

    python -m mobileposer.realtime.run --checkpoint results/ours_multi_finetune/layouts/wrists_shanks_waist/model_finetuned.pth --record recordings/standing.jsonl

The recording uses Python host receive/process timestamps because the current
SolarXR feed does not expose a sensor sample timestamp. Quaternion order is
xyzw; linear acceleration is gravity-removed m/s2.

Rebuild the feature stream and model output and save a live-vs-replay report:

    python -m mobileposer.realtime.replay_recording recordings/standing.jsonl --report recordings/standing.replay.json

The expected unchanged-path thresholds are featureMaxAbs below 1e-5 and
poseMaxAbs below 1e-4. A/B switches are:

- --heading recompute
- --disable-acceleration
- --disable-orientation
- --broadcast (send the replay to Unity)

Compare calibrated live inputs against an existing processed capture:

    python -m mobileposer.realtime.compare_inputs recordings/standing.jsonl results/ours_multi_finetune/captures/CAPTURE/inputs/wrists_shanks_waist/ours1.pt --output recordings/standing.distribution.json

In Unity, enable Show Pose Diagnostics on MobilePoserPoseSource to display the
output frame, SolarXR update index, declared four-frame model lag, host timing,
and approximate transport age. This display does not alter the pose.
