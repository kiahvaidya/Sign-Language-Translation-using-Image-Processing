"""
Webcam Processor Module
Processes captured live webcam frames, handles targeted hand framing,
isolates the hand from room surroundings, and translates gestures.
"""

import cv2
import numpy as np

from translation_service import TranslationService
from hand_segmentation import detect_hand_bounding_box
from image_preprocessing import load_image, encode_image_base64


class WebcamProcessor:
    """
    Handles live/snapshot webcam inputs.
    Applies hand framing guides, forearm pruning, and translation inference.
    """

    def __init__(self):
        self.service = TranslationService()

    def process_frame(self, frame_source, framing_mode="target_box", box_preset="right_hand", box_size=240):
        """
        Processes single camera frame:
        - framing_mode: 'target_box', 'auto_detect', or 'full_frame'
        - box_preset: 'right_hand', 'center_hand', 'left_hand'
        """
        img_bgr, orig_rgb, orig_gray, orig_hsv, metadata = load_image(frame_source)
        h, w = img_bgr.shape[:2]

        # 1. Determine bounding box for hand isolation
        if framing_mode == "target_box":
            if box_preset == "right_hand":
                cx, cy = int(w * 0.68), int(h * 0.42)
            elif box_preset == "left_hand":
                cx, cy = int(w * 0.32), int(h * 0.42)
            else:  # Center
                cx, cy = int(w * 0.50), int(h * 0.45)

            half = box_size // 2
            x1 = max(0, cx - half)
            y1 = max(0, cy - half)
            x2 = min(w, cx + half)
            y2 = min(h, cy + half)

        elif framing_mode == "auto_detect":
            x1, y1, bw, bh = detect_hand_bounding_box(img_bgr)
            x2, y2 = x1 + bw, y1 + bh

        else:  # Full frame
            x1, y1, x2, y2 = 0, 0, w, h

        # 2. Viewfinder overlay for user visual feedback
        viewfinder = img_bgr.copy()
        cv2.rectangle(viewfinder, (x1, y1), (x2, y2), (0, 240, 100), 2)
        cv2.putText(viewfinder, "HAND TARGET (CROPPED)", (x1 + 4, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 240, 100), 2)

        # 3. Crop hand and run DIP translation
        hand_crop = img_bgr[y1:y2, x1:x2]
        if hand_crop.size == 0:
            hand_crop = img_bgr

        res = self.service.translate_single_image(hand_crop, prune_wrist=True)
        res["viewfinder_overlay"] = encode_image_base64(viewfinder, is_bgr=True)
        res["crop_bbox"] = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]

        return res
