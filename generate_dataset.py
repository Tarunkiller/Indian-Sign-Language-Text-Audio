"""
Synthetic ISL Dataset Generator
================================
Generates plausible 63-dimensional hand-landmark feature vectors for
each ISL class (A-Z, 0-9) by simulating MediaPipe Hands output.

Each finger is modelled as 4 joints (MCP → PIP → DIP → TIP).
A state value in [0,1] controls extension:  0 = fully curled, 1 = fully open.

Run:  python data/generate_dataset.py
Output: data/isl_keypoints.csv
"""

import numpy as np
import pandas as pd
import os

# ── Finger state definitions ──────────────────────────────────────────────────
# [thumb, index, middle, ring, pinky]  0=closed  1=open  0.5=half-bent
ISL_STATES = {
    # Letters
    "A": [0.3, 0.0, 0.0, 0.0, 0.0],
    "B": [0.0, 1.0, 1.0, 1.0, 1.0],
    "C": [0.5, 0.5, 0.5, 0.5, 0.4],
    "D": [0.0, 1.0, 0.2, 0.2, 0.2],
    "E": [0.0, 0.2, 0.2, 0.2, 0.2],
    "F": [0.6, 0.2, 1.0, 1.0, 1.0],
    "G": [0.8, 1.0, 0.0, 0.0, 0.0],
    "H": [0.4, 1.0, 1.0, 0.0, 0.0],
    "I": [0.0, 0.0, 0.0, 0.0, 1.0],
    "J": [0.0, 0.0, 0.0, 0.0, 0.9],
    "K": [0.8, 1.0, 1.0, 0.0, 0.0],
    "L": [1.0, 1.0, 0.0, 0.0, 0.0],
    "M": [0.0, 0.1, 0.1, 0.1, 0.0],
    "N": [0.0, 0.1, 0.1, 0.0, 0.0],
    "O": [0.4, 0.4, 0.4, 0.4, 0.4],
    "P": [0.8, 1.0, 1.0, 0.0, 0.0],
    "Q": [0.8, 1.0, 0.0, 0.0, 0.0],
    "R": [0.0, 1.0, 1.0, 0.0, 0.0],
    "S": [0.2, 0.0, 0.0, 0.0, 0.0],
    "T": [0.4, 0.2, 0.0, 0.0, 0.0],
    "U": [0.0, 1.0, 1.0, 0.0, 0.0],
    "V": [0.0, 1.0, 1.0, 0.0, 0.0],
    "W": [0.0, 1.0, 1.0, 1.0, 0.0],
    "X": [0.0, 0.5, 0.0, 0.0, 0.0],
    "Y": [1.0, 0.0, 0.0, 0.0, 1.0],
    "Z": [0.0, 1.0, 0.0, 0.0, 0.0],
    # Digits
    "0": [0.4, 0.4, 0.4, 0.4, 0.4],
    "1": [0.0, 1.0, 0.0, 0.0, 0.0],
    "2": [0.0, 1.0, 1.0, 0.0, 0.0],
    "3": [0.6, 1.0, 1.0, 0.0, 0.0],
    "4": [0.0, 1.0, 1.0, 1.0, 1.0],
    "5": [1.0, 1.0, 1.0, 1.0, 1.0],
    "6": [1.0, 0.0, 0.0, 0.0, 1.0],
    "7": [1.0, 0.0, 0.0, 1.0, 0.0],
    "8": [1.0, 0.0, 1.0, 0.0, 0.0],
    "9": [0.4, 0.3, 0.0, 0.0, 0.0],
}

# ── Anatomical landmark templates ────────────────────────────────────────────
# 21 landmarks × (x,y,z) relative to wrist (landmark 0 = origin).
# Values are approximate in a unit-square space; y↑ = up.

# Base positions when hand is fully OPEN (all fingers extended)
OPEN_LANDMARKS = np.array([
    # Wrist
    [0.00,  0.00,  0.00],
    # Thumb: CMC, MCP, IP, TIP
    [0.12,  0.05,  0.00],
    [0.20,  0.10,  0.00],
    [0.28,  0.02, -0.01],
    [0.34, -0.07, -0.02],
    # Index: MCP, PIP, DIP, TIP
    [0.10, -0.20,  0.00],
    [0.10, -0.38,  0.00],
    [0.10, -0.52,  0.00],
    [0.10, -0.65,  0.00],
    # Middle: MCP, PIP, DIP, TIP
    [0.00, -0.22,  0.00],
    [0.00, -0.40,  0.00],
    [0.00, -0.54,  0.00],
    [0.00, -0.68,  0.00],
    # Ring: MCP, PIP, DIP, TIP
    [-0.10, -0.20,  0.00],
    [-0.10, -0.38,  0.00],
    [-0.10, -0.52,  0.00],
    [-0.10, -0.64,  0.00],
    # Pinky: MCP, PIP, DIP, TIP
    [-0.18, -0.16,  0.00],
    [-0.20, -0.30,  0.00],
    [-0.20, -0.42,  0.00],
    [-0.20, -0.52,  0.00],
], dtype=np.float32)

# Positions when hand is fully CLOSED (fist)
CLOSED_LANDMARKS = np.array([
    [0.00,  0.00,  0.00],
    [0.10,  0.03,  0.02],
    [0.15,  0.00,  0.04],
    [0.12, -0.06,  0.06],
    [0.08, -0.12,  0.04],
    [0.10, -0.20,  0.00],
    [0.08, -0.24,  0.05],
    [0.06, -0.20,  0.08],
    [0.05, -0.16,  0.08],
    [0.00, -0.22,  0.00],
    [-0.01, -0.26,  0.05],
    [-0.02, -0.22,  0.08],
    [-0.02, -0.18,  0.08],
    [-0.10, -0.20,  0.00],
    [-0.10, -0.24,  0.05],
    [-0.10, -0.21,  0.08],
    [-0.10, -0.17,  0.08],
    [-0.18, -0.16,  0.00],
    [-0.18, -0.19,  0.04],
    [-0.18, -0.17,  0.07],
    [-0.18, -0.14,  0.07],
], dtype=np.float32)

# Finger landmark indices: [MCP, PIP, DIP, TIP]
FINGER_INDICES = {
    "thumb":  [1, 2, 3, 4],
    "index":  [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "pinky":  [17, 18, 19, 20],
}
FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]


def state_to_landmarks(finger_states: list) -> np.ndarray:
    """Interpolate between open/closed poses based on finger state values."""
    lm = OPEN_LANDMARKS.copy()
    for fi, finger in enumerate(FINGER_ORDER):
        state = finger_states[fi]   # 0=closed, 1=open
        for idx in FINGER_INDICES[finger]:
            lm[idx] = (
                state * OPEN_LANDMARKS[idx] + (1 - state) * CLOSED_LANDMARKS[idx]
            )
    return lm.flatten()   # 63-dim


def generate_dataset(samples_per_class: int = 800, noise_std: float = 0.025):
    rows = []
    for label, states in ISL_STATES.items():
        base = state_to_landmarks(states)
        for _ in range(samples_per_class):
            noisy = base + np.random.normal(0, noise_std, base.shape)
            rows.append([label] + noisy.tolist())

    cols = ["label"] + [f"f{i}" for i in range(63)]
    df = pd.DataFrame(rows, columns=cols)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


if __name__ == "__main__":
    out_dir = os.path.dirname(__file__)
    out_path = os.path.join(out_dir, "isl_keypoints.csv")
    print("Generating ISL synthetic dataset ...")
    df = generate_dataset(samples_per_class=800)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} samples -> {out_path}")
    print(f"Classes ({len(df['label'].unique())}): {sorted(df['label'].unique())}")
