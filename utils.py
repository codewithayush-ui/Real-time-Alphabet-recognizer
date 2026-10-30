import numpy as np

def extract_hand_features(landmarks):
    """
    Extracts 21 3D landmarks (63 values), centers them at the wrist,
    and scales them to [-1, 1] range for scale and position invariance.
    """
    # Extract x, y, z for all 21 points
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

    # 1. Translate wrist (index 0) to origin (0, 0, 0)
    wrist = coords[0]
    translated = coords - wrist

    # 2. Scale invariance: divide by the maximum coordinate magnitude
    max_distance = np.max(np.abs(translated))
    if max_distance == 0:
        max_distance = 1e-6

    normalized = translated / max_distance

    # Flatten 21 * 3 matrix into a 1D array of 63 features
    return normalized.flatten().tolist()