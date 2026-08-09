"""
Joint Angle Calculator – computes flexion angles for all 15 finger joints.

Uses the law of cosines on the two bone vectors meeting at each joint.
Result: 15 angles (3 per finger) in degrees (0°–180°).
"""

import math

# Each definition: (name, proximal_index, center_index, distal_index).
# These use MediaPipe hand landmark indices (0–20), where 0 is the wrist
# and fingers are arranged thumb→index→middle→ring→little.
# The angle is computed at the center landmark for each joint.
JOINT_ANGLE_DEFS: list[tuple[str, int, int, int]] = [
    # Thumb (wrist→CMC→MP→IP→Tip = indices 0,1,2,3,4)
    ("thumb_cmc",  0, 1, 2),
    ("thumb_mp",   1, 2, 3),
    ("thumb_ip",   2, 3, 4),
    # Index (wrist→MCP→PIP→DIP→Tip = 0,5,6,7,8)
    ("index_mcp",  0, 5, 6),
    ("index_pip",  5, 6, 7),
    ("index_dip",  6, 7, 8),
    # Middle (0,9,10,11,12)
    ("middle_mcp", 0,  9, 10),
    ("middle_pip", 9, 10, 11),
    ("middle_dip", 10, 11, 12),
    # Ring (0,13,14,15,16)
    ("ring_mcp",   0, 13, 14),
    ("ring_pip",  13, 14, 15),
    ("ring_dip",  14, 15, 16),
    # Little (0,17,18,19,20)
    ("little_mcp",  0, 17, 18),
    ("little_pip", 17, 18, 19),
    ("little_dip", 18, 19, 20),
]


def compute_joint_angles(landmarks) -> dict[str, float]:
    """Compute all 15 flexion angles from MediaPipe landmarks.

    Args:
        landmarks: MediaPipe NormalizedLandmarkList (list-like, 21 items).

    Returns:
        Dict mapping angle name (e.g. "index_pip") to degrees (0–180).
    """
    result: dict[str, float] = {}

    for name, prox_i, cent_i, dist_i in JOINT_ANGLE_DEFS:
        pA = landmarks[prox_i]
        pB = landmarks[cent_i]
        pC = landmarks[dist_i]

        # Build the two bone vectors that meet at the center joint.
        # BA points from the center joint toward the proximal point.
        # BC points from the center joint toward the distal point.
        ba_x = pA.x - pB.x
        ba_y = pA.y - pB.y
        bc_x = pC.x - pB.x
        bc_y = pC.y - pB.y

        dot = ba_x * bc_x + ba_y * bc_y
        mag_ba = math.hypot(ba_x, ba_y)
        mag_bc = math.hypot(bc_x, bc_y)

        # Skip unstable or degenerate joint configurations.
        if mag_ba < 1e-9 or mag_bc < 1e-9:
            continue

        # Use the cosine law to convert the vector dot product into an angle.
        cos_angle = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
        degrees = math.degrees(math.acos(cos_angle))
        result[name] = round(degrees, 1)

    return result
