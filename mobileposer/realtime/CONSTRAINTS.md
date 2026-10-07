# Runtime constraints

The MobilePoser Unity receiver keeps all post-processing disabled by default.
The browser viewer can change the optional switches while the local pose stream
is running. Open `http://127.0.0.1:8765/`, expand `Runtime constraints`, and
change one option at a time. Settings are sent over the existing pose WebSocket
and are applied by `MobilePoserPoseSource`.

Available controls:

- `Tracker anchors`: optional measured tracker direction fusion.
- `Joint limits`: clamp avatar rotations relative to the bind pose.
- `Knee guard`: prevent near-straight knee hyperextension.
- `Temporal smoothing`: smooth incoming model rotations.
- `Bone lengths`: enable the measured avatar chain solver.
- `Knee direction`: use the calibrated neutral-pose knee plane as the IK pole.
- `Contact foot IK`: solve a contacting hip-knee-ankle chain toward the floor.
- `Foot grounding`: calibrate and maintain the relative floor plane.

Recommended A/B order:

1. Enable `Temporal smoothing` only.
2. Add `Knee direction` and `Bone lengths`.
3. Enable `Foot grounding`, then `Contact foot IK`.
4. Test `Joint limits`, `Knee guard`, and `Tracker anchors` separately.

The controls change the Unity post-processing only; they do not alter the
MobilePose checkpoint or the Python model output. Foot IK requires a calibrated
relative floor and contact probabilities. If the Python stream does not include
foot contact, that constraint remains inactive.
