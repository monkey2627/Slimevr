# MobilePoser Hardware Test Runbook

This is the practical runbook for the first physical test. The pipeline is:

`IMU -> SlimeVR-Server -> SolarXR -> Python mobileposer -> Unity`

The test runs on one PC. Unity is run directly in the Editor; no Unity build
or packaged application is required.

## 0. Preflight

1. Start the desktop SlimeVR-Server.
2. Connect all five IMUs, assign the intended SlimeVR BodyParts, and complete
   SlimeVR's normal pose calibration. This is separate from the startup
   calibration performed by `run.py`.
3. Do not change tracker bindings, reset trackers, or recalibrate SlimeVR while
   this run is in progress. If the tracker set changes, restart `run.py`.
4. Use an environment containing `torch`, `numpy`, `scipy`, `tqdm`,
   `flatbuffers`, and `websockets`. The known environment is
   `E:\dyh\conda_envs\mobileposer-realtime`.
5. If Torch was compiled against NumPy 1.x but NumPy 2.x is installed, fix the
   environment before running:

   ```powershell
   python -m pip install "numpy<2"
   python -c "import numpy, torch; print(numpy.__version__, torch.__version__); print(torch.tensor([1.0]).numpy())"
   ```

6. Confirm the matching checkpoint exists under
   `checkpoints/no_head_5imu_surface/<layout>/1/base_model.pth`. The six
   supported layouts are defined by `mobileposer/realtime/layout.py`.

## 1. Verify SolarXR

From the repository root:

```powershell
cd E:\dyh\MotionRecover\code\base_mobileposer
python -m mobileposer.realtime.probe_solarxr
```

Expected behavior:

- The client connects to `ws://127.0.0.1:21110`.
- Five online, mappable trackers are listed with BodyPart, status, quaternion,
  and acceleration.
- Moving an IMU changes its corresponding row.

If this step cannot connect or does not show changing data, stop here and fix
SlimeVR/SolarXR first. Stop the probe with `Ctrl+C`.

## 2. Run inference and startup calibration

In a second PowerShell window:

```powershell
cd E:\dyh\MotionRecover\code\base_mobileposer
python -m mobileposer.realtime.run
```

Expected output sequence:

1. `Resolved layout: ...` after the online tracker set exactly matches a
   trained layout. A near match is not accepted; the process waits.
2. `Calibrating: rotate/move...` for the default 150 frames, about five
   seconds.
3. One `yaw=... residual_p95=...deg` line per tracker.
4. `Calibrated. Streaming live pose...` and broadcast on
   `ws://127.0.0.1:21200`.

During the calibration window, all five trackers in the detected layout must
make clear, continuous rotations. Use a mixed motion such as twisting the
torso, swinging both arms, and alternating leg lifts or knee bends. The motion
does not need to be fast or precise, but do not leave one tracker still.

Calibration errors are retried automatically:

- `No detectable rotation`: that tracker's window contained too little usable
  rotation. Move that body part more clearly in the next window.
- `Inconsistent raw<->reference-adjusted relationship`: motion was present but
  the raw and adjusted rotations were not consistent with one fixed heading.
  Keep all trackers stable, avoid resets/recalibration, and repeat with smooth
  multi-axis motion.

If a previous process is already using port 21200, stop it before starting a
new `run.py`. A tracker set change also requires restarting `run.py`; it has no
hot swap.

## 3. Test Unity in the Editor

Open `IMUTrack-for-Spine` in Unity `2022.3.62f3` (see
`ProjectSettings/ProjectVersion.txt`). Create or open a dedicated scene that
contains `Assets/Models/mesh_edit1.vrm`. (There is no checked-in scene named
`AvatarScene.unity`.) Attach
`Assets/slimeVR/Scripts/MobilePoserPoseSource.cs` to the avatar GameObject that
contains the `Animator`. Assign the `animator` field in the Inspector if it is
not found automatically, and leave `Retarget Through Global Bind Pose` enabled.
Do not test on an avatar that is simultaneously driven by EVMC4U
`ExternalReceiver`, `VmcTrackerPoseSource`, an Animator Controller, pose
playback, or another script that writes Humanoid bones; use a clean duplicate
of the avatar or disable those pose sources first.

After Python prints `Calibrated. Streaming live pose...`, press **Play** in the
Unity Editor. The script connects to `ws://127.0.0.1:21200`; do not build a
standalone Unity application for this test.

### Unity-only retargeting check

To isolate Unity from SolarXR and the neural network, replay a processed
MobilePoser `.pt` file containing a `pose` sequence:

```powershell
python -m mobileposer.realtime.replay_pose `
  data/no_head_5imu_surface_dip_test/no_head_5imu_surface_dip_test/ `
  wrists_shanks_waist/dip_test.pt `
  --sequence 0 --loop
```

This starts the same local WebSocket server on `21200` and sends stored SMPL
local rotations to Unity. Stop `run.py` first so the replay server can bind the
port.
If this replay is also twisted or has reversed hip/shoulder directions, the
problem is in the Unity Avatar retargeting or another Unity component writing
the bones. If replay is correct but live `run.py` is wrong, investigate the
SolarXR layout, calibration, and live model input instead.

To replay checkpoint predictions from the same `acc` and `ori` data:

```powershell
python -m mobileposer.realtime.replay_infer `
  data/no_head_5imu_surface_dip_test/no_head_5imu_surface_dip_test/ `
  wrists_shanks_waist/dip_test.pt --sequence 0 --loop
```

This applies the training normalization (`acc / 30`), runs the 45-frame
`MobilePose` window, and broadcasts predicted SMPL local rotations. The
matching checkpoint is selected from
`checkpoints/no_head_5imu_surface/<layout>/1/base_model.pth`.

To simulate the live adapter from a recorded SolarXR raw-IMU package:

```powershell
python -m mobileposer.realtime.replay_raw_imu `
  "data/raw/ours/10_站立位伸展/1/10_v1.raw-imu.json" `
  --layout wrists_shanks_waist --loop
```

This reconstructs the SolarXR values from `orientation`,
`rotationReferenceAdjusted`, and `accelerationG * 9.80665`, then uses the same
heading calibration, feature builder, checkpoint, and Unity broadcaster as
`run.py`. Because the saved action does not include a separate startup
calibration gesture, its complete motion is used to fit the fixed heading.

In this project, an ankle-mounted tracker means the distal-shank `ls`/`rs`
attachment (`LEFT_LOWER_LEG`/`RIGHT_LOWER_LEG` in SlimeVR), so it uses
`wrists_shanks_waist`. The separate `lf`/`rf` attachment is on the dorsal foot
and maps to `LEFT_FOOT`/`RIGHT_FOOT`; it is not the ankle placement used here.

Test: raise both arms, twist the torso, bend the knees, and take a few steps.
Record overall left/right and up/down direction, and any abnormal shoulder,
hip, or other joint. The current stream drives pose rotations only; it does
not provide root translation.

## 4. Evidence to save for the next session

- Python environment, Torch version, and NumPy version.
- Detected layout and its five BodyParts.
- Each calibration `residual_p95` value.
- Whether `Calibrated. Streaming live pose...` appeared.
- Unity scene, connection result, overall direction, and abnormal joint names.
- Complete Python terminal output and Unity Console output.

Do not regenerate or overwrite the checked-in `solarxr_protocol` bindings
casually; they include a manual fix for a `flatc --gen-all` Python import bug.
SolarXR acceleration is already in `m/s^2`; do not multiply it by 9.80665.
