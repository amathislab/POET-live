"""
Utilities for keypoints manipulation.
"""

import torch

COCO_CLASSES = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

SKELETON = [ [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],[6,8],
            [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7] ]


def kpts_xyxy_to_cxcydxdy(kpts, ctr, hierarchical=False):
    kpts = build_SPR(kpts, ctr)
    return torch.cat((ctr, kpts), dim=1)


# Structured Pose Representation (SPR)
def build_SPR(keypoints, center):
    keypoints[:, 0::3] = keypoints[:, 0::3] - center[:, 0].unsqueeze_(1)
    keypoints[:, 1::3] = keypoints[:, 1::3] - center[:, 1].unsqueeze_(1)
    return keypoints

