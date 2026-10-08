using SpineFlow.MobilePoser;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

internal static class MobilePoserSmplInstaller
{
    private const string AvatarScenePath = "Assets/Scenes/AvatarScene.unity";
    private const string ModelPath =
        "Assets/Resources/SMPL/SMPL_m_unityDoubleBlends_lbs_10_scale5_207_v1.0.0.fbx";
    private const string InstanceName = "MobilePoser SMPL Reference";
    private static readonly string[] RequiredBones =
    {
        "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee",
        "Spine2", "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot",
        "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
        "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand",
    };

    [MenuItem("Tools/MobilePoser/Add Direct SMPL Reference")]
    public static void AddDirectSmplReference()
    {
        GameObject model = AssetDatabase.LoadAssetAtPath<GameObject>(ModelPath);
        if (model == null)
        {
            EditorUtility.DisplayDialog("MobilePoser", "SMPL model is missing:\n" + ModelPath, "OK");
            return;
        }

        GameObject existing = GameObject.Find(InstanceName);
        if (existing != null)
        {
            Selection.activeGameObject = existing;
            EditorGUIUtility.PingObject(existing);
            EditorUtility.DisplayDialog("MobilePoser",
                "The current scene already contains a direct SMPL reference.", "OK");
            return;
        }

        GameObject instance = (GameObject)PrefabUtility.InstantiatePrefab(model);
        Undo.RegisterCreatedObjectUndo(instance, "Add direct SMPL reference");
        instance.name = InstanceName;
        instance.transform.SetPositionAndRotation(new Vector3(1.2f, 0f, 0f), Quaternion.identity);
        ValidateRig(instance);
        instance.AddComponent<MobilePoserSmplPoseSource>();
        Selection.activeGameObject = instance;
        EditorGUIUtility.PingObject(instance);
        EditorSceneManager.MarkSceneDirty(instance.scene);
        Debug.Log("Added direct MobilePose SMPL reference. It connects to ws://127.0.0.1:21200.",
            instance);
    }

    public static void InstallIntoAvatarScene()
    {
        var scene = EditorSceneManager.OpenScene(AvatarScenePath, OpenSceneMode.Single);
        AddDirectSmplReference();
        if (!EditorSceneManager.SaveScene(scene))
            throw new System.InvalidOperationException("Failed to save " + AvatarScenePath);
    }

    private static void ValidateRig(GameObject instance)
    {
        Transform[] descendants = instance.GetComponentsInChildren<Transform>(true);
        foreach (string suffix in RequiredBones)
        {
            bool found = false;
            foreach (Transform candidate in descendants)
            {
                if (candidate.name == suffix || candidate.name.EndsWith("_" + suffix,
                        System.StringComparison.OrdinalIgnoreCase))
                {
                    found = true;
                    break;
                }
            }
            if (!found)
            {
                Object.DestroyImmediate(instance);
                throw new System.InvalidOperationException("SMPL FBX is missing bone " + suffix);
            }
        }
    }
}
