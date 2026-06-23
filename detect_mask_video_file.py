# USAGE
# python detect_mask_video_file.py --input path/to/video.mp4 --output path/to/output.mp4

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os


def detect_and_predict_mask(frame, face_net, mask_net, confidence):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []
    locs = []
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
        locs.append((startX, startY, endX, endY))

    preds = []
    if faces:
        preds = mask_net.predict(np.array(faces, dtype="float32"), batch_size=32, verbose=0)

    annotated = frame.copy()
    for (startX, startY, endX, endY), (mask_prob, no_mask_prob) in zip(locs, preds):
        label = "Mask" if mask_prob > no_mask_prob else "No Mask"
        prob = max(mask_prob, no_mask_prob)
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        text = f"{label}: {prob * 100:.2f}%"
        cv2.putText(annotated, text, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(annotated, (startX, startY), (endX, endY), color, 2)

    return annotated, len(locs)


def main():
    ap = argparse.ArgumentParser(description="Run mask detection on a video file")
    ap.add_argument("-i", "--input", required=True, help="path to input video")
    ap.add_argument("-o", "--output", default=None, help="path to save annotated video")
    ap.add_argument("-f", "--face", type=str, default="face_detector",
                    help="path to face detector model directory")
    ap.add_argument("-m", "--model", type=str, default="mask_detector.model",
                    help="path to trained mask detector model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="minimum face detection confidence")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="optional limit on how many seconds to process")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Video not found: {args.input}")

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_predicted{ext or '.mp4'}"

    print("[INFO] loading face detector model...")
    prototxt = os.path.sep.join([args.face, "deploy.prototxt"])
    weights = os.path.sep.join([args.face, "res10_300x300_ssd_iter_140000.caffemodel"])
    face_net = cv2.dnn.readNet(prototxt, weights)

    print("[INFO] loading mask detector model...")
    mask_net = load_model(args.model)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = int(args.max_seconds * fps) if args.max_seconds else None

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_idx = 0
    frames_with_faces = 0
    total_faces = 0

    print(f"[INFO] processing {args.input}")
    while True:
        if max_frames is not None and frame_idx >= max_frames:
            break

        ok, frame = cap.read()
        if not ok:
            break

        annotated, face_count = detect_and_predict_mask(
            frame, face_net, mask_net, args.confidence
        )
        writer.write(annotated)
        if face_count:
            frames_with_faces += 1
            total_faces += face_count
        frame_idx += 1

        if frame_idx % 50 == 0:
            print(f"[INFO] processed {frame_idx} frames...")

    cap.release()
    writer.release()

    print(f"[INFO] done — {frame_idx} frames, {frames_with_faces} with faces, {total_faces} total detections")
    print(f"[INFO] saved to {args.output}")


if __name__ == "__main__":
    main()
