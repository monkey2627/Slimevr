# Runtime constraints

The MobilePoser Unity receiver keeps post-processing disabled by default.
During Play mode, use the `MobilePoser runtime constraints` panel in the Unity
Game view. The browser viewer remains for stick-figure display, recording,
replay, and input comparison; it does not control Unity constraints.

Available controls:

- `Tracker anchors`: optional measured tracker direction fusion.
- `Joint limits`: clamp avatar rotations relative to the bind pose.
- `Knee guard`: prevent near-straight knee hyperextension.
- `Temporal smoothing`: smooth incoming model rotations.
- `Bone lengths`: enable the measured avatar chain solver.
- `Knee direction`: use the calibrated neutral-pose knee plane as the IK pole.
- `Contact foot IK`: solve a contacting hip-knee-ankle chain toward the floor.
- `Foot grounding`: calibrate and maintain the relative floor plane.
- `Calibrate T-pose`: capture a T-pose baseline and map it to the Avatar's
  cached startup/reset rotations.
- `Calibrate relaxed arms`: record the complete natural-standing model
  baseline. Unity uses the raw-model correction for the whole body near this
  baseline and blends toward the T-pose correction only as the shoulders move
  toward the T-pose baseline.

Recommended A/B order:

1. Click `Reset`, stand with relaxed arms, then click `Calibrate relaxed arms`;
   remain still until capture completes.
2. Stand in T-pose and click `Calibrate T-pose`; remain still until capture
   completes.
3. Enable `Temporal smoothing` only.
4. Add `Knee direction` and `Bone lengths`.
5. Enable `Foot grounding`, then `Contact foot IK`.
6. Test `Joint limits`, `Knee guard`, and `Tracker anchors` separately.

The controls change Unity post-processing only; they do not alter the MobilePose
checkpoint or Python model output. The two calibrations are rotation offsets,
not a pose estimator: they reduce mismatch between T-pose and natural-standing
outputs but do not guarantee arbitrary live poses. Foot IK requires a calibrated relative floor and
contact probabilities. If the Python stream does not include foot contact,
that constraint remains inactive.
