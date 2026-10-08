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
    /// Drives the official SMPL Unity rig directly from MobilePose's 24 local
    /// joint rotations. This intentionally bypasses Humanoid retargeting and
    /// provides a reference for diagnosing Avatar-specific axis errors.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class MobilePoserSmplPoseSource : MonoBehaviour
    {
        [SerializeField] private Transform skeletonRoot;
        [SerializeField] private string host = "127.0.0.1";
        [SerializeField] private int port = 21200;
        [SerializeField] private bool applyReceivedPose = true;
        [SerializeField] private bool multiplyBindRotation = false;
        [SerializeField] private bool showDiagnostics = true;

        private const int JointCount = 24;

        private static readonly string[] JointNames =
        {
            "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee",
            "Spine2", "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot",
            "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
            "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand",
        };

        private readonly Transform[] bones = new Transform[JointCount];
        private readonly Quaternion[] bindLocalRotations = new Quaternion[JointCount];
        private readonly object sync = new object();
        private CancellationTokenSource cancellation;
        private Task worker;
        private float[] latestJoints;
        private int latestFrame = -1;
        private string status = "Waiting for MobilePose...";

        public string CurrentStatus
        {
            get { lock (sync) return status; }
        }

        private void Awake()
        {
            if (skeletonRoot == null) skeletonRoot = transform;
        }

        private void Start()
        {
            if (!CacheBones())
            {
                enabled = false;
                return;
            }
            cancellation = new CancellationTokenSource();
            worker = Task.Run(() => RunConnectionLoopAsync(cancellation.Token));
        }

        private void OnDestroy()
        {
            cancellation?.Cancel();
            cancellation?.Dispose();
        }

        private bool CacheBones()
        {
            Transform[] descendants = skeletonRoot.GetComponentsInChildren<Transform>(true);
            int missing = 0;
            for (int i = 0; i < JointCount; i++)
            {
                string suffix = JointNames[i];
                foreach (Transform candidate in descendants)
                {
                    if (candidate.name == suffix || candidate.name.EndsWith("_" + suffix,
                            StringComparison.OrdinalIgnoreCase))
                    {
                        bones[i] = candidate;
                        bindLocalRotations[i] = candidate.localRotation;
                        break;
                    }
                }
                if (bones[i] == null)
                {
                    missing++;
                    Debug.LogError($"MobilePoserSmplPoseSource: missing SMPL bone {suffix}.", this);
                }
            }
            if (missing > 0)
            {
                SetStatus($"SMPL rig invalid: {missing} bones missing");
                return false;
            }
            SetStatus("SMPL rig ready; connecting...");
            return true;
        }

        private void LateUpdate()
        {
            if (!applyReceivedPose) return;
            float[] joints;
            lock (sync) joints = latestJoints;
            if (joints == null || joints.Length != JointCount * 4) return;

            for (int i = 0; i < JointCount; i++)
            {
                int offset = i * 4;
                var rotation = new Quaternion(joints[offset], joints[offset + 1],
                    joints[offset + 2], joints[offset + 3]);
                float magnitudeSquared = rotation.x * rotation.x + rotation.y * rotation.y +
                    rotation.z * rotation.z + rotation.w * rotation.w;
                if (!float.IsNaN(magnitudeSquared) && !float.IsInfinity(magnitudeSquared) &&
                    magnitudeSquared > 0.000001f)
                {
                    rotation.Normalize();
                    bones[i].localRotation = multiplyBindRotation
                        ? bindLocalRotations[i] * rotation
                        : rotation;
                }
            }
        }

        private void OnGUI()
        {
            if (!showDiagnostics) return;
            GUI.Box(new Rect(12f, Screen.height - 82f, 420f, 68f), string.Empty);
            GUI.Label(new Rect(24f, Screen.height - 74f, 396f, 52f),
                $"MobilePose direct SMPL\nFrame: {latestFrame}  {CurrentStatus}");
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
                        SetStatus("Connected (direct SMPL)");
                        await ReceiveLoopAsync(socket, token).ConfigureAwait(false);
                    }
                }
                catch (OperationCanceledException) when (token.IsCancellationRequested)
                {
                    return;
                }
                catch (Exception exception)
                {
                    SetStatus("Connection failed: " + exception.Message);
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

                    if (result.MessageType != WebSocketMessageType.Text) continue;
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
                return;
            }
            if (frame?.joints == null || frame.joints.Length != JointCount * 4) return;
            lock (sync)
            {
                latestJoints = frame.joints;
                latestFrame = frame.frame;
            }
        }

        private void SetStatus(string value)
        {
            lock (sync) status = value;
        }

        [Serializable]
        private sealed class PoseFrame
        {
            public int frame;
            public float[] joints;
        }
    }
}
