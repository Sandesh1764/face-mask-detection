"""Process user videos: full annotated output + random frame snapshots."""
import json
import os
import random

import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

VIDEOS = [
    {
        "input": "KakaoTalk_20260623_111820946.mp4",
        "label": "with_mask",
        "description": "User video — person wearing mask",
    },
    {
        "input": "KakaoTalk_20260623_111832880.mp4",
        "label": "without_mask",
        "description": "User video — person without mask",
    },
]

OUTPUT_ROOT = "user_predictions"
RANDOM_FRAMES = 6
CONFIDENCE = 0.5
SEED = 42


def load_networks():
    face_net = cv2.dnn.readNet(
        "face_detector/deploy.prototxt",
        "face_detector/res10_300x300_ssd_iter_140000.caffemodel",
    )
    mask_net = load_model("mask_detector.model")
    return face_net, mask_net


def predict_frame(frame, face_net, mask_net, confidence=0.5):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    faces, locs = [], []
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf <= confidence:
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
        locs.append((startX, startY, endX, endY, float(conf)))

    preds = []
    if faces:
        preds = mask_net.predict(np.array(faces, dtype="float32"), verbose=0)

    annotated = frame.copy()
    results = []
    for (startX, startY, endX, endY, face_conf), (mask_prob, no_mask_prob) in zip(locs, preds):
        label = "Mask" if mask_prob > no_mask_prob else "No Mask"
        prob = float(max(mask_prob, no_mask_prob))
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        text = f"{label}: {prob * 100:.2f}%"
        cv2.putText(annotated, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(annotated, (startX, startY), (endX, endY), color, 2)
        results.append({
            "label": label,
            "confidence": prob,
            "mask_prob": float(mask_prob),
            "no_mask_prob": float(no_mask_prob),
            "face_detection_confidence": face_conf,
        })

    return annotated, results


def process_video(video_info, face_net, mask_net, rng):
    input_path = video_info["input"]
    stem = os.path.splitext(os.path.basename(input_path))[0]
    out_dir = os.path.join(OUTPUT_ROOT, stem)
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    video_out = os.path.join(out_dir, f"{stem}_predicted.mp4")
    writer = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frames_with_faces = []
    frame_idx = 0
    summary = {
        "input": input_path,
        "expected": video_info["label"],
        "description": video_info["description"],
        "output_video": video_out,
        "frames_dir": frames_dir,
        "total_frames": total_frames,
        "frames_processed": 0,
        "frames_with_faces": 0,
        "prediction_counts": {"Mask": 0, "No Mask": 0},
        "saved_random_frames": [],
    }

    print(f"[INFO] processing {input_path} ({total_frames} frames)")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        annotated, results = predict_frame(frame, face_net, mask_net, CONFIDENCE)
        writer.write(annotated)
        summary["frames_processed"] += 1

        if results:
            summary["frames_with_faces"] += 1
            frames_with_faces.append((frame_idx, annotated.copy(), results))

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  ... {frame_idx}/{total_frames} frames")

    cap.release()
    writer.release()

    # Save random snapshot frames (prefer frames that had face detections)
    pool = frames_with_faces if frames_with_faces else []
    if pool:
        picks = rng.sample(pool, min(RANDOM_FRAMES, len(pool)))
        for idx, (frame_num, image, results) in enumerate(picks, start=1):
            out_img = os.path.join(frames_dir, f"random_{idx:02d}_frame_{frame_num:05d}.jpg")
            cv2.imwrite(out_img, image)
            labels = [r["label"] for r in results]
            for r in results:
                summary["prediction_counts"][r["label"]] += 1
            summary["saved_random_frames"].append({
                "frame": frame_num,
                "path": out_img,
                "predictions": results,
                "labels": labels,
            })

    # Normalize prediction counts to per-frame picks only; recompute from all frames
    summary["prediction_counts"] = {"Mask": 0, "No Mask": 0}
    mask_frames = 0
    nomask_frames = 0
    for _, _, results in frames_with_faces:
        labels = [r["label"] for r in results]
        if all(l == "Mask" for l in labels):
            mask_frames += 1
        elif all(l == "No Mask" for l in labels):
            nomask_frames += 1
        for r in results:
            summary["prediction_counts"][r["label"]] += 1

    summary["dominant_prediction"] = (
        "Mask" if summary["prediction_counts"]["Mask"] > summary["prediction_counts"]["No Mask"]
        else "No Mask" if summary["prediction_counts"]["No Mask"] > summary["prediction_counts"]["Mask"]
        else "mixed"
    )
    print(f"[INFO] saved video -> {video_out}")
    print(f"[INFO] saved {len(summary['saved_random_frames'])} random frames -> {frames_dir}")
    return summary


def main():
    random.seed(SEED)
    rng = random.Random(SEED)

    print("[INFO] loading models...")
    face_net, mask_net = load_networks()

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    report = {"videos": []}

    for video in VIDEOS:
        if not os.path.exists(video["input"]):
            raise FileNotFoundError(video["input"])
        report["videos"].append(process_video(video, face_net, mask_net, rng))

    report_path = os.path.join(OUTPUT_ROOT, "predictions.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"[INFO] report -> {report_path}")


if __name__ == "__main__":
    main()
