"""Read display rotation from video metadata and correct frame orientation."""
import json
import subprocess

import cv2

_ROTATE_CV = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def get_video_rotation(path):
    """Return clockwise display-rotation degrees from container metadata (0 if none)."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream_tags=rotate",
        "-of",
        "json",
        path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        if not streams:
            return 0
        rotate = streams[0].get("tags", {}).get("rotate")
        return int(rotate) if rotate is not None else 0
    except (subprocess.CalledProcessError, json.JSONDecodeError, ValueError, FileNotFoundError):
        return 0


def orient_frame(frame, rotation_degrees):
    """Rotate raw decoded frame so it matches the intended display orientation."""
    if rotation_degrees not in _ROTATE_CV:
        return frame
    return cv2.rotate(frame, _ROTATE_CV[rotation_degrees])
