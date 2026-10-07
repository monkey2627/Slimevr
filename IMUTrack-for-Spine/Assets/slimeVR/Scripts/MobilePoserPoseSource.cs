using System;
using System.IO;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace SpineFlow.MobilePoser
{
    /// <summary>
    /// Connects to the local mobileposer realtime bridge
    /// (mobileposer/realtime/run.py + stream_out.py, ws://127.0.0.1:21200 by
    /// default) and drives a Humanoid Animator's bones directly from its SMPL
    /// pose stream. This is an alternative pose source alongside the existing
    /// SlimeVR VMC/EVMC4U path, not a replacement for it -- SlimeVR-Server
    /// keeps running unmodified, so both can be compared side by side (see the
    /// implementation plan's M6).
    ///
    /// Attach to the GameObject holding the Animator for the avatar to drive
    /// (e.g. wherever mesh_edit1.vrm's Animator lives), or assign `animator`
    /// explicitly in the inspector.
    /// </summary>
    public sealed class MobilePoserPoseSource : MonoBehaviour
    {
        [SerializeField] private Animator animator;
        [SerializeField] private string host = "127.0.0.1";
        [SerializeField] private int port = 21200;
        [Tooltip("Keep receiving diagnostics but do not apply the incoming pose to the Avatar.")]
        [SerializeField] private bool applyReceivedPose = true;
        [Tooltip("Retarget SMPL rotations through world-space animation deltas. Keep this " +
                 "enabled for VRM/Humanoid avatars: their per-bone local axes do not in " +
                 "general match SMPL's local axes.")]
        [SerializeField] private bool retargetThroughGlobalBindPose = true;
        [Tooltip("Blend measured tracker directions back into the model for the five observed body parts.")]
        [SerializeField] private bool useTrackerAnchors = false;
        [Range(0f, 1f)]
        [SerializeField] private float trackerAnchorWeight = 0.65f;
        [SerializeField] private float trackerAnchorMaxAngle = 75f;
        [Tooltip("Clamp driven bones relative to the Avatar bind pose. Verify the Avatar before enabling.")]
        [SerializeField] private bool enforceJointLimits = false;
        [SerializeField] private float maxLegRotation = 170f;
        [SerializeField] private float maxArmRotation = 165f;
        [SerializeField] private float maxFootRotation = 90f;
        [SerializeField] private bool guardKneeHyperextension = false;
        [SerializeField] private float minimumKneeAngle = 3f;
        [Tooltip("Smooth incoming model rotations before applying them to the Avatar.")]
        [SerializeField] private bool useTemporalSmoothing = false;
        [Range(0f, 1f)]
        [SerializeField] private float temporalSmoothing = 0.35f;
        [Tooltip("Keep two-bone leg and arm chains at their measured Avatar lengths.")]
        [SerializeField] private bool enforceBoneLengths = false;
        [Tooltip("Use the neutral-pose knee plane as a stable pole direction.")]
        [SerializeField] private bool useKneeDirectionConstraint = false;
        [SerializeField] private float kneeDirectionWeight = 0.65f;
        [Tooltip("Solve contacting feet with a two-bone hip/knee/ankle IK chain.")]
        [SerializeField] private bool useFootIk = false;
        [SerializeField] private float footIkWeight = 0.8f;
        [SerializeField] private float footIkMaxReach = 0.03f;
        [SerializeField] private bool showFootContactDebug = false;
        [Tooltip("Show incoming frame/timestamp/model-lag diagnostics without changing the pose.")]
        [SerializeField] private bool showPoseDiagnostics = false;
        [Tooltip("Show runtime constraint switches in the Unity Game view.")]
        [SerializeField] private bool showConstraintControls = false;
        [Tooltip("Calibrate a relative floor from the initial contacting feet and correct the avatar root.")]
        [SerializeField] private bool useRelativeFootGrounding = false;
        [SerializeField] private float contactEngageThreshold = 0.8f;
        [SerializeField] private float contactReleaseThreshold = 0.5f;
        [SerializeField] private int groundingCalibrationFrames = 15;
        [SerializeField] private float maxGroundCorrectionPerFrame = 0.03f;
        [SerializeField] private float maxGroundDropPerFrame = 0.015f;
        [SerializeField] private bool groundWholeBody = true;
        [SerializeField] private float groundContactSlack = 0.015f;
        [SerializeField] private bool showRelativeGroundGuide = false;
        [SerializeField] private float relativeGroundGuideSize = 4f;
        [SerializeField] private Color relativeGroundGuideColor = new Color(0.1f, 0.8f, 1f, 0.18f);
        [SerializeField] private int groundingCheckInterval = 4;
        [SerializeField] private int debugGuiInterval = 4;

        // SMPL joint index -> HumanBodyBones, per mobileposer/fit_ours_smpl.py's
        // joint-index comment: 0 pelvis, 1/2 L/R hip, 3 spine1, 4/5 L/R knee,
        // 6 spine2, 7/8 L/R ankle, 9 spine3, 10/11 L/R foot, 12 neck, 13/14 L/R
        // collar, 15 head, 16/17 L/R shoulder, 18/19 L/R elbow, 20/21 L/R wrist,
        // 22/23 L/R hand. HumanBodyBones has no direct "hand tip" bone distinct
        // from LeftHand/RightHand, so 22/23 are skipped (LastBone sentinel).
        private static readonly HumanBodyBones[] JointToBone =
        {
            HumanBodyBones.Hips,               // 0  pelvis
            HumanBodyBones.LeftUpperLeg,        // 1  L hip
            HumanBodyBones.RightUpperLeg,       // 2  R hip
            HumanBodyBones.Spine,               // 3  spine1
            HumanBodyBones.LeftLowerLeg,        // 4  L knee
            HumanBodyBones.RightLowerLeg,       // 5  R knee
            HumanBodyBones.Chest,               // 6  spine2
            HumanBodyBones.LeftFoot,            // 7  L ankle
            HumanBodyBones.RightFoot,           // 8  R ankle
            HumanBodyBones.UpperChest,          // 9  spine3
            HumanBodyBones.LeftToes,            // 10 L foot
            HumanBodyBones.RightToes,           // 11 R foot
            HumanBodyBones.Neck,                // 12 neck
            HumanBodyBones.LeftShoulder,        // 13 L collar
            HumanBodyBones.RightShoulder,       // 14 R collar
            HumanBodyBones.Head,                // 15 head
            HumanBodyBones.LeftUpperArm,        // 16 L shoulder
            HumanBodyBones.RightUpperArm,       // 17 R shoulder
            HumanBodyBones.LeftLowerArm,        // 18 L elbow
            HumanBodyBones.RightLowerArm,       // 19 R elbow
            HumanBodyBones.LeftHand,            // 20 L wrist
            HumanBodyBones.RightHand,           // 21 R wrist
            HumanBodyBones.LastBone,            // 22 L hand -- no distinct Humanoid bone
            HumanBodyBones.LastBone,            // 23 R hand -- no distinct Humanoid bone
        };

        // Standard 24-joint SMPL kinematic tree. The received matrices are
        // local rotations expressed in each SMPL parent's frame.
        private static readonly int[] JointParents =
        {
            -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
             9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21,
        };

        private const int JointCount = 24;

        private readonly Transform[] boneTransforms = new Transform[JointCount];
        private readonly Quaternion[] restLocalRotation = new Quaternion[JointCount];
        private readonly Quaternion[] restWorldRotation = new Quaternion[JointCount];
        private readonly Quaternion[] smplWorldRotation = new Quaternion[JointCount];

        private CancellationTokenSource cancellation;
        private Task worker;
        private readonly object sync = new object();
        private float[] latestJoints; // flat, JointCount * 4, guarded by sync
        private float[] latestFootContact; // optional [left, right] probabilities
        private float[] latestTrackerRotations; // optional five tracker quaternions, layout order
        private int latestFrame = -1;
        private int latestInputFrame = -1;
        private int latestModelLagFrames = -1;
        private double latestHostUnixSeconds;
        private double latestHostMonotonicSeconds;
        private double latestUnityReceiveRealtime;
        private double latestUnityApplyRealtime;
        private int groundingStableFrames;
        private bool relativeGroundCalibrated;
        private bool relativeGroundActive;
        private float relativeGroundY;
        private Renderer[] bodyRenderers;
        private GameObject relativeGroundGuide;
        private int groundingCheckCounter;
        private int debugGuiCounter;
        private string debugGuiText = "MobilePoser foot contact";
        private string lastError = "Connecting to mobileposer realtime bridge...";
        private readonly Quaternion[] trackerAnchorOffsets = new Quaternion[5];
        private readonly Quaternion[] trackerAnchorMeasured = new Quaternion[5];
        private bool trackerAnchorsCalibrated;
        private readonly float[] boneLengths = new float[JointCount];
        private readonly Quaternion[] smoothedSmplWorldRotation = new Quaternion[JointCount];
        private bool hasSmoothedPose;
        private Vector3 leftKneePole;
        private Vector3 rightKneePole;

        public string CurrentStatus
        {
            get { lock (sync) return lastError ?? "Streaming mobileposer pose"; }
        }

        public float LeftFootContact => GetFootContact(0);
        public float RightFootContact => GetFootContact(1);

        private float GetFootContact(int index)
        {
            lock (sync)
            {
                return latestFootContact != null && latestFootContact.Length > index
                    ? latestFootContact[index] : -1f;
            }
        }

        private void OnGUI()
        {
            if (!showFootContactDebug && !showConstraintControls && !showPoseDiagnostics) return;
            if (showConstraintControls)
                DrawConstraintControls();
            if (showPoseDiagnostics)
                DrawPoseDiagnostics();
            if (!showFootContactDebug) return;
            // Do not pay the IMGUI cost during the normal pose-only stream.
            // Older Unity scenes may have this serialized toggle enabled even
            // when Python is not sending the optional footContact field.
            lock (sync)
            {
                if (latestFootContact == null) return;
            }
            if (++debugGuiCounter >= Mathf.Max(1, debugGuiInterval))
            {
                debugGuiCounter = 0;
                float left = LeftFootContact;
                float right = RightFootContact;
                string leftText = left < 0f ? "n/a" : left.ToString("0.000");
                string rightText = right < 0f ? "n/a" : right.ToString("0.000");
                string floor = relativeGroundCalibrated ? relativeGroundY.ToString("0.000") : "calibrating";
                debugGuiText = $"MobilePoser foot contact\nLeft: {leftText}   Right: {rightText}\nFrame: {latestFrame}  floor: {floor}";
            }
            GUI.Box(new Rect(12f, 12f, 360f, 92f), string.Empty);
            GUI.Label(new Rect(24f, 20f, 340f, 70f), debugGuiText);
        }

        private void DrawPoseDiagnostics()
        {
            int frame;
            int inputFrame;
            int lagFrames;
            double hostUnix;
            double hostMonotonic;
            double received;
            double applied;
            lock (sync)
            {
                frame = latestFrame;
                inputFrame = latestInputFrame;
                lagFrames = latestModelLagFrames;
                hostUnix = latestHostUnixSeconds;
                hostMonotonic = latestHostMonotonicSeconds;
                received = latestUnityReceiveRealtime;
                applied = latestUnityApplyRealtime;
            }
            double transportAgeMs = hostUnix > 0.0
                ? Math.Max(0.0, (DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() / 1000.0 - hostUnix) * 1000.0)
                : -1.0;
            string text = frame < 0
                ? "MobilePoser pose diagnostics\nWaiting for a pose frame..."
                : $"MobilePoser pose diagnostics\nOutput frame: {frame}  Source update: {inputFrame}\n" +
                  $"Model lag: {(lagFrames < 0 ? "n/a" : lagFrames + " frames")}  Host mono: {hostMonotonic:0.000}\n" +
                  $"Transport age: {(transportAgeMs < 0 ? "n/a" : transportAgeMs.ToString("0.0") + " ms")}  " +
                  $"Receive/apply: {received:0.000} / {applied:0.000}";
            GUI.Box(new Rect(384f, 12f, 390f, 94f), string.Empty);
            GUI.Label(new Rect(396f, 20f, 370f, 80f), text);
        }

        private void DrawConstraintControls()
        {
            const float x = 12f;
            const float y = 116f;
            const float width = 360f;
            GUI.Box(new Rect(x, y, width, 300f), "MobilePoser constraints");
            useTrackerAnchors = GUI.Toggle(new Rect(x + 12f, y + 24f, width - 24f, 22f),
                useTrackerAnchors, "Tracker anchors");
            enforceJointLimits = GUI.Toggle(new Rect(x + 12f, y + 48f, width - 24f, 22f),
                enforceJointLimits, "Joint rotation limits");
            guardKneeHyperextension = GUI.Toggle(new Rect(x + 12f, y + 72f, width - 24f, 22f),
                guardKneeHyperextension, "Knee hyperextension guard");
            useTemporalSmoothing = GUI.Toggle(new Rect(x + 12f, y + 96f, width - 24f, 22f),
                useTemporalSmoothing, "Temporal rotation smoothing");
            enforceBoneLengths = GUI.Toggle(new Rect(x + 12f, y + 120f, width - 24f, 22f),
                enforceBoneLengths, "Bone length constraint");
            useKneeDirectionConstraint = GUI.Toggle(new Rect(x + 12f, y + 144f, width - 24f, 22f),
                useKneeDirectionConstraint, "Knee direction constraint");
            useFootIk = GUI.Toggle(new Rect(x + 12f, y + 168f, width - 24f, 22f),
                useFootIk, "Contact foot IK");
            useRelativeFootGrounding = GUI.Toggle(new Rect(x + 12f, y + 192f, width - 24f, 22f),
                useRelativeFootGrounding, "Foot grounding");
            showRelativeGroundGuide = GUI.Toggle(new Rect(x + 12f, y + 216f, width - 24f, 22f),
                showRelativeGroundGuide, "Ground guide");
            GUI.Label(new Rect(x + 12f, y + 242f, width - 24f, 20f),
                $"Anchor {trackerAnchorWeight:0.00}  Smooth {temporalSmoothing:0.00}  IK {footIkWeight:0.00}");
            trackerAnchorWeight = GUI.HorizontalSlider(new Rect(x + 12f, y + 262f, 105f, 18f),
                trackerAnchorWeight, 0f, 1f);
            temporalSmoothing = GUI.HorizontalSlider(new Rect(x + 124f, y + 262f, 105f, 18f),
                temporalSmoothing, 0f, 1f);
            footIkWeight = GUI.HorizontalSlider(new Rect(x + 236f, y + 262f, 105f, 18f),
                footIkWeight, 0f, 1f);
        }

        private void Awake()
        {
            if (animator == null) animator = GetComponent<Animator>();
        }

        private void Start()
        {
            CacheBoneTransformsAndRestPose();
            bodyRenderers = GetComponentsInChildren<Renderer>(true);
            cancellation = new CancellationTokenSource();
            worker = Task.Run(() => RunConnectionLoopAsync(cancellation.Token));
        }

        private void OnDestroy()
        {
            if (relativeGroundGuide != null) Destroy(relativeGroundGuide);
            cancellation?.Cancel();
            cancellation?.Dispose();
        }

        private void CacheBoneTransformsAndRestPose()
        {
            if (animator == null)
            {
                Debug.LogError("MobilePoserPoseSource: no Animator found/assigned.", this);
                return;
            }
            for (int i = 0; i < JointCount; i++)
            {
                if (JointToBone[i] == HumanBodyBones.LastBone) continue;
                Transform t = animator.GetBoneTransform(JointToBone[i]);
                boneTransforms[i] = t;
                if (t != null)
                {
                    restLocalRotation[i] = t.localRotation;
                    restWorldRotation[i] = t.rotation;
                }
            }
            CacheBoneLengths();
            leftKneePole = KneePoleDirection(1, 4, 7);
            rightKneePole = KneePoleDirection(2, 5, 8);
        }

        private void CacheBoneLengths()
        {
            int[] children = { -1, 4, 5, -1, 7, 8, -1, 10, 11, -1, -1, -1,
                -1, -1, -1, -1, 18, 19, 20, 21, -1, -1, -1, -1 };
            for (int i = 0; i < JointCount; i++)
            {
                Transform parent = boneTransforms[i];
                Transform child = children[i] >= 0 ? boneTransforms[children[i]] : null;
                boneLengths[i] = parent != null && child != null
                    ? Vector3.Distance(parent.position, child.position) : 0f;
            }
            Debug.Log($"MobilePoser bone lengths: L thigh={boneLengths[1]:F3}, L shin={boneLengths[4]:F3}, " +
                      $"R thigh={boneLengths[2]:F3}, R shin={boneLengths[5]:F3}, " +
                      $"L upper arm={boneLengths[16]:F3}, L forearm={boneLengths[18]:F3}, " +
                      $"R upper arm={boneLengths[17]:F3}, R forearm={boneLengths[19]:F3}", this);
        }

        private void LateUpdate()
        {
            float[] joints;
            lock (sync)
            {
                joints = latestJoints;
            }
            if (joints == null || joints.Length != JointCount * 4) return;
            if (!applyReceivedPose) return;

            // Reconstruct SMPL global rotations before retargeting. Directly
            // multiplying avatar local rotations by SMPL local rotations is
            // invalid for a general VRM rig because their bind axes differ.
            for (int i = 0; i < JointCount; i++)
            {
                int o = i * 4;
                float x = joints[o], y = joints[o + 1], z = joints[o + 2], w = joints[o + 3];
                float magnitudeSquared = x * x + y * y + z * z + w * w;
                if (float.IsNaN(magnitudeSquared) || float.IsInfinity(magnitudeSquared) ||
                    magnitudeSquared < 0.000001f)
                {
                    smplWorldRotation[i] = i == 0
                        ? Quaternion.identity
                        : smplWorldRotation[JointParents[i]];
                    continue;
                }

                var local = new Quaternion(x, y, z, w);
                local.Normalize();
                int parent = JointParents[i];
                smplWorldRotation[i] = parent < 0
                    ? local
                    : smplWorldRotation[parent] * local;
            }
            if (useTemporalSmoothing)
            {
                float blend = Mathf.Clamp01(1f - temporalSmoothing);
                for (int i = 0; i < JointCount; i++)
                {
                    smoothedSmplWorldRotation[i] = !hasSmoothedPose
                        ? smplWorldRotation[i]
                        : Quaternion.Slerp(smoothedSmplWorldRotation[i], smplWorldRotation[i], blend);
                }
                hasSmoothedPose = true;
            }
            else
            {
                for (int i = 0; i < JointCount; i++) smoothedSmplWorldRotation[i] = smplWorldRotation[i];
                hasSmoothedPose = false;
            }
            for (int i = 0; i < JointCount; i++)
            {
                Transform t = boneTransforms[i];
                if (t == null) continue;

                if (retargetThroughGlobalBindPose)
                {
                    // targetGlobal = animationDeltaGlobal * targetRestGlobal.
                    t.rotation = smoothedSmplWorldRotation[i] * restWorldRotation[i];
                }
                else
                {
                    // Diagnostic legacy path only, for an exactly SMPL-aligned rig.
                    int o = i * 4;
                    var local = new Quaternion(joints[o], joints[o + 1], joints[o + 2], joints[o + 3]);
                    local.Normalize();
                    t.localRotation = restLocalRotation[i] * local;
                }
            }

            ApplyTrackerAnchors();
            ApplyJointConstraints();
            ApplyKinematicConstraints();
            if (showRelativeGroundGuide && relativeGroundCalibrated)
                CreateOrUpdateRelativeGroundGuide();
            else if (!showRelativeGroundGuide && relativeGroundGuide != null)
                relativeGroundGuide.SetActive(false);

            if (++groundingCheckCounter >= Mathf.Max(1, groundingCheckInterval))
            {
                groundingCheckCounter = 0;
                ApplyRelativeFootGrounding();
            }
            lock (sync)
            {
                latestUnityApplyRealtime = System.Diagnostics.Stopwatch.GetTimestamp() /
                    (double)System.Diagnostics.Stopwatch.Frequency;
            }
        }

        private void ApplyTrackerAnchors()
        {
            if (!useTrackerAnchors || animator == null) return;
            float[] trackerValues;
            lock (sync) trackerValues = latestTrackerRotations;
            if (trackerValues == null || trackerValues.Length < 20) return;

            // wrists, shanks, waist -> SMPL/Unity humanoid bones.
            int[] trackerBones = { 20, 21, 4, 5, 0 };
            for (int i = 0; i < 5; i++)
            {
                int o = i * 4;
                trackerAnchorMeasured[i] = new Quaternion(trackerValues[o], trackerValues[o + 1],
                    trackerValues[o + 2], trackerValues[o + 3]);
                if (Quaternion.Dot(trackerAnchorMeasured[i], trackerAnchorMeasured[i]) < 0.000001f) return;
                trackerAnchorMeasured[i].Normalize();
            }

            if (!trackerAnchorsCalibrated)
            {
                for (int i = 0; i < 5; i++)
                {
                    Transform bone = boneTransforms[trackerBones[i]];
                    if (bone == null) return;
                    trackerAnchorOffsets[i] = Quaternion.Inverse(trackerAnchorMeasured[i]) * bone.rotation;
                }
                trackerAnchorsCalibrated = true;
                return;
            }

            float weight = Mathf.Clamp01(trackerAnchorWeight);
            for (int i = 0; i < 5; i++)
            {
                Transform bone = boneTransforms[trackerBones[i]];
                if (bone == null) continue;
                Quaternion target = trackerAnchorMeasured[i] * trackerAnchorOffsets[i];
                float angle = Quaternion.Angle(bone.rotation, target);
                if (angle > trackerAnchorMaxAngle && trackerAnchorMaxAngle > 0f)
                    target = Quaternion.Slerp(bone.rotation, target, trackerAnchorMaxAngle / angle);
                bone.rotation = Quaternion.Slerp(bone.rotation, target, weight);
            }
        }

        private void ApplyJointConstraints()
        {
            if (!enforceJointLimits) return;
            ClampWorldRotations(new[] { 1, 2, 4, 5 }, maxLegRotation);
            ClampWorldRotations(new[] { 16, 17, 18, 19 }, maxArmRotation);
            ClampWorldRotations(new[] { 7, 8, 10, 11 }, maxFootRotation);
            if (guardKneeHyperextension)
            {
                GuardKneeAngle(1, 4, 7);
                GuardKneeAngle(2, 5, 8);
            }
        }

        private void ApplyKinematicConstraints()
        {
            if (enforceBoneLengths || useKneeDirectionConstraint)
            {
                SolveLegChain(1, 4, 7, leftKneePole, useKneeDirectionConstraint ? kneeDirectionWeight : 0f);
                SolveLegChain(2, 5, 8, rightKneePole, useKneeDirectionConstraint ? kneeDirectionWeight : 0f);
            }
            if (useFootIk)
            {
                float left = LeftFootContact;
                float right = RightFootContact;
                if (left >= contactEngageThreshold) SolveContactFoot(1, 4, 7, 10, left);
                if (right >= contactEngageThreshold) SolveContactFoot(2, 5, 8, 11, right);
            }
        }

        private Vector3 KneePoleDirection(int hipIndex, int kneeIndex, int ankleIndex)
        {
            Transform hip = boneTransforms[hipIndex];
            Transform knee = boneTransforms[kneeIndex];
            Transform ankle = boneTransforms[ankleIndex];
            if (hip == null || knee == null || ankle == null) return Vector3.forward;
            Vector3 planeNormal = Vector3.Cross(knee.position - hip.position, ankle.position - knee.position);
            if (planeNormal.sqrMagnitude < 0.000001f) return animator.transform.forward;
            Vector3 pole = Vector3.Cross(planeNormal.normalized, (knee.position - hip.position).normalized);
            return pole.sqrMagnitude < 0.000001f ? animator.transform.forward : pole.normalized;
        }

        private void SolveLegChain(int hipIndex, int kneeIndex, int ankleIndex, Vector3 pole, float poleWeight)
        {
            Transform hip = boneTransforms[hipIndex];
            Transform knee = boneTransforms[kneeIndex];
            Transform ankle = boneTransforms[ankleIndex];
            if (hip == null || knee == null || ankle == null) return;
            Vector3 hipPosition = hip.position;
            Vector3 target = ankle.position;
            SolveTwoBone(hip, knee, ankle, target, pole, poleWeight, footIkMaxReach);
            if (enforceBoneLengths)
            {
                Vector3 delta = ankle.position - target;
                if (delta.sqrMagnitude > 0.000001f) animator.transform.position -= delta;
            }
            if (hip.position != hipPosition) animator.transform.position += hipPosition - hip.position;
        }

        private void SolveContactFoot(int hipIndex, int kneeIndex, int ankleIndex, int footIndex, float contact)
        {
            Transform foot = boneTransforms[footIndex] != null ? boneTransforms[footIndex] : boneTransforms[ankleIndex];
            if (foot == null || !relativeGroundCalibrated) return;
            Vector3 target = foot.position;
            target.y = relativeGroundY;
            float weight = Mathf.Clamp01(footIkWeight * contact);
            target = Vector3.Lerp(foot.position, target, weight);
            SolveTwoBone(boneTransforms[hipIndex], boneTransforms[kneeIndex], boneTransforms[ankleIndex],
                target, hipIndex == 1 ? leftKneePole : rightKneePole,
                kneeDirectionWeight * weight, footIkMaxReach);
            Quaternion levelRotation = Quaternion.FromToRotation(foot.up, Vector3.up) * foot.rotation;
            foot.rotation = Quaternion.Slerp(foot.rotation, levelRotation, weight);
        }

        private static void SolveTwoBone(Transform hip, Transform knee, Transform ankle, Vector3 target,
            Vector3 pole, float poleWeight, float reachSlack)
        {
            if (hip == null || knee == null || ankle == null) return;
            Vector3 a = hip.position;
            float upper = Vector3.Distance(a, knee.position);
            float lower = Vector3.Distance(knee.position, ankle.position);
            if (upper < 0.0001f || lower < 0.0001f) return;
            Vector3 toTarget = target - a;
            float distance = Mathf.Clamp(toTarget.magnitude, Mathf.Abs(upper - lower) + 0.0001f,
                upper + lower - Mathf.Max(0.0001f, reachSlack));
            Vector3 direction = toTarget.sqrMagnitude < 0.000001f ? hip.forward : toTarget.normalized;
            Vector3 poleDirection = Vector3.ProjectOnPlane(pole, direction);
            if (poleDirection.sqrMagnitude < 0.000001f) poleDirection = Vector3.Cross(direction, Vector3.up);
            poleDirection.Normalize();
            float cosAngle = Mathf.Clamp((upper * upper + distance * distance - lower * lower) /
                (2f * upper * distance), -1f, 1f);
            float along = upper * cosAngle;
            float height = upper * Mathf.Sqrt(Mathf.Max(0f, 1f - cosAngle * cosAngle));
            Vector3 desiredKnee = a + direction * along + poleDirection * height;
            Quaternion hipDelta = Quaternion.FromToRotation(knee.position - a, desiredKnee - a);
            hip.rotation = hipDelta * hip.rotation;
            Vector3 newAnkle = ankle.position;
            Quaternion kneeDelta = Quaternion.FromToRotation(newAnkle - knee.position, target - knee.position);
            knee.rotation = kneeDelta * knee.rotation;
            if (poleWeight > 0f)
            {
                Vector3 currentPole = Vector3.ProjectOnPlane(knee.position - a, direction).normalized;
                Quaternion poleDelta = Quaternion.FromToRotation(currentPole, poleDirection);
                hip.rotation = Quaternion.Slerp(hip.rotation, poleDelta * hip.rotation, Mathf.Clamp01(poleWeight));
            }
        }

        private void ClampWorldRotations(int[] indices, float maxAngle)
        {
            if (maxAngle <= 0f) return;
            foreach (int index in indices)
            {
                Transform bone = boneTransforms[index];
                if (bone == null) continue;
                float angle = Quaternion.Angle(restWorldRotation[index], bone.rotation);
                if (angle > maxAngle)
                    bone.rotation = Quaternion.Slerp(restWorldRotation[index], bone.rotation, maxAngle / angle);
            }
        }

        private void GuardKneeAngle(int hipIndex, int kneeIndex, int ankleIndex)
        {
            Transform hip = boneTransforms[hipIndex];
            Transform knee = boneTransforms[kneeIndex];
            Transform ankle = boneTransforms[ankleIndex];
            if (hip == null || knee == null || ankle == null) return;
            float angle = Vector3.Angle(hip.position - knee.position, ankle.position - knee.position);
            if (angle < minimumKneeAngle)
                knee.rotation = Quaternion.Slerp(restWorldRotation[kneeIndex], knee.rotation, 0.5f);
        }

        private void ApplyRelativeFootGrounding()
        {
            if (!useRelativeFootGrounding || animator == null) return;
            float leftContact = LeftFootContact;
            float rightContact = RightFootContact;
            if (leftContact < 0f || rightContact < 0f) return;
            Transform leftFoot = boneTransforms[10] != null ? boneTransforms[10] : boneTransforms[7];
            Transform rightFoot = boneTransforms[11] != null ? boneTransforms[11] : boneTransforms[8];
            if (leftFoot == null || rightFoot == null) return;

            float lowest = groundWholeBody ? LowestBodyRendererY() : Mathf.Min(leftFoot.position.y, rightFoot.position.y);
            if (float.IsInfinity(lowest)) return;
            bool bothContacting = leftContact >= contactEngageThreshold && rightContact >= contactEngageThreshold;
            if (!relativeGroundCalibrated)
            {
                groundingStableFrames = bothContacting ? groundingStableFrames + 1 : 0;
                if (groundingStableFrames >= Mathf.Max(1, groundingCalibrationFrames))
                {
                    relativeGroundY = lowest;
                    relativeGroundCalibrated = true;
                    relativeGroundActive = true;
                    CreateOrUpdateRelativeGroundGuide();
                }
                return;
            }

            if (groundWholeBody)
            {
                // After a standing calibration, keep a one-sided upward
                // penetration guard even when both feet leave the floor.
                relativeGroundActive = true;
            }
            else if (!relativeGroundActive)
                relativeGroundActive = leftContact >= contactEngageThreshold || rightContact >= contactEngageThreshold;
            else if (leftContact < contactReleaseThreshold && rightContact < contactReleaseThreshold)
                relativeGroundActive = false;
            if (!relativeGroundActive) return;

            // Follow the relative plane in both directions, but limit the rate
            // and ignore tiny renderer-bound jitter. This lets a lying pose
            // settle onto the plane without a sudden whole-body jump.
            float error = relativeGroundY - lowest;
            if (Mathf.Abs(error) <= groundContactSlack) return;
            float correction = error > 0f
                ? Mathf.Min(error - groundContactSlack, maxGroundCorrectionPerFrame)
                : Mathf.Max(error + groundContactSlack, -maxGroundDropPerFrame);
            animator.transform.position += Vector3.up * correction;
        }

        private void CreateOrUpdateRelativeGroundGuide()
        {
            if (!showRelativeGroundGuide || !relativeGroundCalibrated || animator == null) return;
            if (relativeGroundGuide == null)
            {
                relativeGroundGuide = GameObject.CreatePrimitive(PrimitiveType.Plane);
                relativeGroundGuide.name = "MobilePoser Relative Ground Guide";
                Collider collider = relativeGroundGuide.GetComponent<Collider>();
                if (collider != null) Destroy(collider);
                MeshRenderer renderer = relativeGroundGuide.GetComponent<MeshRenderer>();
                Shader shader = Shader.Find("Unlit/Color");
                if (shader == null) shader = Shader.Find("Universal Render Pipeline/Unlit");
                if (shader == null) shader = Shader.Find("Sprites/Default");
                Material material = new Material(shader);
                material.color = relativeGroundGuideColor;
                renderer.material = material;
            }
            if (!relativeGroundGuide.activeSelf)
            {
                relativeGroundGuide.transform.position = new Vector3(
                    animator.transform.position.x, relativeGroundY, animator.transform.position.z);
                relativeGroundGuide.transform.localScale = Vector3.one * (relativeGroundGuideSize / 10f);
                relativeGroundGuide.SetActive(true);
            }
        }

        private float LowestBodyRendererY()
        {
            if (bodyRenderers == null || bodyRenderers.Length == 0) return float.PositiveInfinity;
            float lowest = float.PositiveInfinity;
            foreach (Renderer renderer in bodyRenderers)
            {
                if (renderer != null && renderer.enabled)
                    lowest = Mathf.Min(lowest, renderer.bounds.min.y);
            }
            return lowest;
        }

        private async Task RunConnectionLoopAsync(CancellationToken token)
        {
            string url = $"ws://{host}:{port}";
            while (!token.IsCancellationRequested)
            {
                try
                {
                    using (var socket = new ClientWebSocket())
                    {
                        await socket.ConnectAsync(new Uri(url), token).ConfigureAwait(false);
                        SetError(null);
                        await ReceiveLoopAsync(socket, token).ConfigureAwait(false);
                    }
                }
                catch (OperationCanceledException) when (token.IsCancellationRequested)
                {
                    return;
                }
                catch (Exception exception)
                {
                    SetError("mobileposer realtime bridge connection failed: " + exception.Message);
                }

                try
                {
                    await Task.Delay(1000, token).ConfigureAwait(false);
                }
                catch (OperationCanceledException)
                {
                    return;
                }
            }
        }

        private async Task ReceiveLoopAsync(ClientWebSocket socket, CancellationToken token)
        {
            var buffer = new byte[65536];
            while (!token.IsCancellationRequested && socket.State == WebSocketState.Open)
            {
                using (var message = new MemoryStream())
                {
                    WebSocketReceiveResult result;
                    do
                    {
                        result = await socket.ReceiveAsync(new ArraySegment<byte>(buffer), token)
                            .ConfigureAwait(false);
                        if (result.MessageType == WebSocketMessageType.Close) return;
                        message.Write(buffer, 0, result.Count);
                    } while (!result.EndOfMessage);

                    if (result.MessageType == WebSocketMessageType.Text)
                        ProcessTextMessage(Encoding.UTF8.GetString(message.ToArray()));
                }
            }
        }

        private void ProcessTextMessage(string json)
        {
            PoseFrame frame;
            try
            {
                frame = JsonUtility.FromJson<PoseFrame>(json);
            }
            catch (Exception)
            {
                return; // malformed/partial frame; drop it and keep the connection alive
            }
            if (frame == null) return;
            if (frame.type == "constraints")
            {
                ApplyRemoteConstraints(frame);
                return;
            }
            if (frame.joints == null || frame.joints.Length != JointCount * 4) return;

            lock (sync)
            {
                latestJoints = frame.joints;
                latestFootContact = frame.footContact;
                latestTrackerRotations = frame.trackerRotations;
                latestFrame = frame.frame;
                latestInputFrame = frame.sourceUpdate != 0 || frame.inputFrame == 0
                    ? frame.sourceUpdate : frame.inputFrame;
                latestModelLagFrames = frame.modelLagFrames;
                latestHostUnixSeconds = frame.receiveUnixSeconds != 0.0
                    ? frame.receiveUnixSeconds : frame.hostUnixSeconds;
                latestHostMonotonicSeconds = frame.receiveMonotonicSeconds != 0.0
                    ? frame.receiveMonotonicSeconds : frame.hostMonotonicSeconds;
                latestUnityReceiveRealtime = System.Diagnostics.Stopwatch.GetTimestamp() /
                    (double)System.Diagnostics.Stopwatch.Frequency;
            }
        }

        private void ApplyRemoteConstraints(PoseFrame frame)
        {
            useTrackerAnchors = frame.useTrackerAnchors;
            enforceJointLimits = frame.enforceJointLimits;
            guardKneeHyperextension = frame.guardKneeHyperextension;
            useTemporalSmoothing = frame.useTemporalSmoothing;
            enforceBoneLengths = frame.enforceBoneLengths;
            useKneeDirectionConstraint = frame.useKneeDirectionConstraint;
            useFootIk = frame.useFootIk;
            useRelativeFootGrounding = frame.useRelativeFootGrounding;
            trackerAnchorWeight = Mathf.Clamp01(frame.trackerAnchorWeight);
            temporalSmoothing = Mathf.Clamp01(frame.temporalSmoothing);
            kneeDirectionWeight = Mathf.Clamp01(frame.kneeDirectionWeight);
            footIkWeight = Mathf.Clamp01(frame.footIkWeight);
        }

        private void SetError(string error)
        {
            lock (sync) lastError = error;
        }

        // Wire format: {"frame": int, "layout": str, "joints": [96 floats]}
        // ("joints" is a FLAT array -- JsonUtility cannot deserialize a nested
        // array-of-arrays). Keep this in sync with
        // mobileposer/realtime/stream_out.py::PoseBroadcaster.broadcast --
        // update both sides together if the payload shape ever changes.
        [Serializable]
        private class PoseFrame
        {
            public string type;
            public int frame;
            public string layout;
            public float[] joints;
            public float[] footContact;
            public float[] trackerRotations;
            public int inputFrame;
            public int sourceUpdate;
            public int modelInputFrame;
            public int modelOutputRepresentsFrame;
            public int modelLagFrames = -1;
            public double hostUnixSeconds;
            public double receiveUnixSeconds;
            public double hostMonotonicSeconds;
            public double receiveMonotonicSeconds;
            public double inferenceStartMonotonicSeconds;
            public double inferenceEndMonotonicSeconds;
            public bool useTrackerAnchors;
            public bool enforceJointLimits;
            public bool guardKneeHyperextension;
            public bool useTemporalSmoothing;
            public bool enforceBoneLengths;
            public bool useKneeDirectionConstraint;
            public bool useFootIk;
            public bool useRelativeFootGrounding;
            public float trackerAnchorWeight;
            public float temporalSmoothing;
            public float kneeDirectionWeight;
            public float footIkWeight;
        }
    }
}
