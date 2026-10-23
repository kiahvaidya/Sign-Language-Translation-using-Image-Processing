"""
Video Processor Module
Extracts video frames at configurable sampling rates, applies the DIP translation pipeline,
and performs temporal smoothing to construct ordered sign sequences and English sentences.
"""

from collections import Counter, deque
import cv2

from translation_service import TranslationService, format_sentence


class VideoProcessor:
    """
    Processes video streams and stored files.
    Applies majority-voting temporal stabilization across frames.
    """

    def __init__(self):
        self.service = TranslationService()

    def process_video(self, video_path, frame_step=3, window_size=5):
        """
        Extracts frames, classifies signs, smooths temporal predictions.

        Returns:
            dict containing:
              - translated_sentence (str)
              - sign_timeline (list of dicts with timestamp, sign, confidence)
              - distinct_signs (list of str)
              - total_frames (int)
              - dip_evidence (list of stage visuals from sample frames)
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        raw_preds = []
        smoothed_preds = []
        timeline = []
        dip_evidence = []
        recent_window = deque(maxlen=window_size)

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_step == 0:
                timestamp_sec = round(frame_idx / fps, 2)
                res = self.service.translate_single_image(frame, prune_wrist=True)
                sign = res["predicted_sign"]
                raw_preds.append(sign)

                recent_window.append(sign)
                most_common = Counter(recent_window).most_common(1)[0][0]
                smoothed_preds.append(most_common)

                if res["has_hand"] and most_common not in ("No Hand Detected", "UNKNOWN", "NOTHING"):
                    timeline.append({
                        "frame": frame_idx,
                        "timestamp": f"{timestamp_sec:.2f}s",
                        "sign": most_common,
                        "confidence": res["confidence"]
                    })

                # Retain sample frames for lab verification
                if len(dip_evidence) < 5 and res["has_hand"]:
                    dip_evidence.append({
                        "frame_index": frame_idx,
                        "timestamp": f"{timestamp_sec:.2f}s",
                        "predicted_sign": most_common,
                        "confidence": res["confidence"],
                        "stages": res["stages"],
                        "metrics": res["metrics"],
                        "metadata": res["metadata"]
                    })

            frame_idx += 1

        cap.release()

        # Deduplicate consecutive tokens
        distinct_tokens = []
        for s in smoothed_preds:
            if not distinct_tokens or distinct_tokens[-1] != s:
                if s not in ("No Hand Detected", "UNKNOWN", "NOTHING", "Model Not Loaded"):
                    distinct_tokens.append(s)

        final_sentence = format_sentence(distinct_tokens)

        return {
            "translated_sentence": final_sentence,
            "distinct_signs": distinct_tokens,
            "timeline": timeline,
            "total_frames": total_frames,
            "processed_frames": frame_idx,
            "dip_evidence": dip_evidence
        }
