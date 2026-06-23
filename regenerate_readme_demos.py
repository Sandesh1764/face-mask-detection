"""Regenerate README demo GIFs and sample images with correct video orientation."""
import glob
import os
import subprocess

import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

from video_orientation import get_video_rotation, orient_frame

MASK_VIDEO = "samples/inputs/videos/KakaoTalk_20260623_111820946.mp4"
NO_MASK_VIDEO = "samples/inputs/videos/KakaoTalk_20260623_111832880.mp4"
CONFIDENCE = 0.5


def load_networks():
    face_net = cv2.dnn.readNet(
        "face_detector/deploy.prototxt",
        "face_detector/res10_300x300_ssd_iter_140000.caffemodel",
    )
    mask_net = load_model("mask_detector.model")
    return face_net, mask_net


def predict_frame(frame, face_net, mask_net):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    faces, locs = [], []
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf <= CONFIDENCE:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        startX, startY = max(0, startX), max(0, startY)
        endX, endY = min(w - 1, endX), min(h - 1, endY)
        face = frame[startY:endY, startX:endX]
        if face.size == 0:
            continue
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        faces.append(face)
        locs.append((startX, startY, endX, endY))

    preds = mask_net.predict(np.array(faces, dtype="float32"), verbose=0) if faces else []

    annotated = frame.copy()
    for (startX, startY, endX, endY), (mask_prob, no_mask_prob) in zip(locs, preds):
        label = "Mask" if mask_prob > no_mask_prob else "No Mask"
        prob = float(max(mask_prob, no_mask_prob))
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        text = f"{label}: {prob * 100:.2f}%"
        cv2.putText(annotated, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(annotated, (startX, startY), (endX, endY), color, 2)

    return annotated, bool(locs)


def collect_oriented_frames(video_path, face_net, mask_net, frame_indices):
    rotation = get_video_rotation(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    wanted = set(frame_indices)
    max_idx = max(frame_indices)
    frames = {}
    idx = 0
    while idx <= max_idx:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in wanted:
            frame = orient_frame(frame, rotation)
            annotated, has_face = predict_frame(frame, face_net, mask_net)
            if has_face:
                frames[idx] = annotated
        idx += 1

    cap.release()
    return [frames[i] for i in frame_indices if i in frames]


def write_clip(frames, output_path, fps=8):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()


def frames_to_gif(frames, output_path, fps=8, width=480):
    tmp_dir = os.path.join("docs", "_gif_frames")
    os.makedirs(tmp_dir, exist_ok=True)
    for old in glob.glob(os.path.join(tmp_dir, "*.jpg")):
        os.remove(old)

    for i, frame in enumerate(frames):
        cv2.imwrite(os.path.join(tmp_dir, f"frame_{i:03d}.jpg"), frame)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pattern = os.path.join(tmp_dir, "frame_%03d.jpg")
    vf = f"fps={fps},scale={width}:-1,split[s0][s1];[s0]palettegen=stats_mode=diff[p];[s1][p]paletteuse=dither=bayer"
    subprocess.run(
        ["ffmpeg", "-y", "-framerate", str(fps), "-i", pattern, "-vf", vf, output_path],
        check=True,
        capture_output=True,
    )
    for old in glob.glob(os.path.join(tmp_dir, "*.jpg")):
        os.remove(old)


def process_mask_demo(face_net, mask_net):
    """Use evenly spaced frames from the mask video for the left README GIF."""
    cap = cv2.VideoCapture(MASK_VIDEO)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    picks = [int(total * p) for p in (0.15, 0.30, 0.45, 0.60, 0.75, 0.90)]
    frames = collect_oriented_frames(MASK_VIDEO, face_net, mask_net, picks)
    if not frames:
        raise RuntimeError("No oriented mask demo frames produced")

    write_clip(frames, "docs/demo_mask_clip.mp4")
    write_clip(frames, "samples/outputs/videos/demo_mask_clip.mp4")
    frames_to_gif(frames, "docs/demo_mask.gif", fps=6, width=480)

    still = frames[len(frames) // 2]
    cv2.imwrite("samples/outputs/images/demo_mask_video_frame.jpg", still)
    print(f"[INFO] mask demo: {len(frames)} frames")


def process_no_mask_demo(face_net, mask_net):
    rotation = get_video_rotation(NO_MASK_VIDEO)
    cap = cv2.VideoCapture(NO_MASK_VIDEO)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {NO_MASK_VIDEO}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    start_frame = int(2 * fps)
    end_frame = start_frame + int(5 * fps)
    frames = []
    idx = 0
    while idx < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if idx >= start_frame and idx % 5 == 0:
            frame = orient_frame(frame, rotation)
            annotated, has_face = predict_frame(frame, face_net, mask_net)
            if has_face:
                frames.append(annotated)
        idx += 1
    cap.release()

    if not frames:
        raise RuntimeError("No oriented no-mask demo frames produced")

    write_clip(frames, "samples/outputs/videos/demo_no_mask_clip.mp4", fps=8)
    frames_to_gif(frames[:20], "docs/demo_no_mask.gif", fps=6, width=400)

    still = frames[len(frames) // 2]
    cv2.imwrite("samples/outputs/images/demo_no_mask_video_frame.jpg", still)
    print(f"[INFO] no-mask demo: {len(frames)} frames")


def main():
    print("[INFO] loading models...")
    face_net, mask_net = load_networks()
    process_mask_demo(face_net, mask_net)
    process_no_mask_demo(face_net, mask_net)
    print("[INFO] done")


if __name__ == "__main__":
    main()
