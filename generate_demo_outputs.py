"""
Generate annotated demo images and video for GitHub.
Only uses media of people wearing or not wearing face masks.
"""
import argparse
import json
import os
import urllib.request

import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

USER_AGENT = "Mozilla/5.0 (compatible; FaceMaskDemo/1.0)"

# People with masks / without masks only (from the upstream Face-Mask-Detection repo)
PUBLIC_IMAGES = [
    {
        "name": "pic1.jpeg",
        "url": "https://raw.githubusercontent.com/chandrikadeb7/Face-Mask-Detection/master/images/pic1.jpeg",
        "description": "Two people wearing surgical masks",
        "source": "Face-Mask-Detection repo (Chandrika Deb)",
        "license": "MIT",
    },
    {
        "name": "out.jpg",
        "url": "https://raw.githubusercontent.com/chandrikadeb7/Face-Mask-Detection/master/images/out.jpg",
        "description": "Three people — mix of mask and no mask",
        "source": "Face-Mask-Detection repo (Chandrika Deb)",
        "license": "MIT",
    },
    {
        "name": "pic2.jpg",
        "url": "https://raw.githubusercontent.com/chandrikadeb7/Face-Mask-Detection/master/images/pic2.jpg",
        "description": "Group of people with mixed mask usage",
        "source": "Face-Mask-Detection repo (Chandrika Deb)",
        "license": "MIT",
    },
    {
        "name": "pic3.jpg",
        "url": "https://raw.githubusercontent.com/chandrikadeb7/Face-Mask-Detection/master/images/pic3.jpg",
        "description": "Single person without a mask",
        "source": "Face-Mask-Detection repo (Chandrika Deb)",
        "license": "MIT",
    },
]

PUBLIC_VIDEO = {
    "name": "people_masks_street.mp4",
    "url": "https://videos.pexels.com/video-files/6833930/6833930-sd_640_360_25fps.mp4",
    "description": "Crowded winter street with masked pedestrians (different people walking)",
    "source": "Pexels — Paweł Wojtasiński (video #6833930)",
    "license": "Pexels License (free for commercial use)",
    "page": "https://www.pexels.com/video/people-walking-on-the-street-during-winter-6833930/",
}


def download_file(url, dest, force=False):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if not force and os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    print(f"[INFO] downloading {url}")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        with open(dest, "wb") as handle:
            handle.write(response.read())
    return dest


def load_networks(face_dir="face_detector", model_path="mask_detector.model"):
    prototxt = os.path.join(face_dir, "deploy.prototxt")
    weights = os.path.join(face_dir, "res10_300x300_ssd_iter_140000.caffemodel")
    face_net = cv2.dnn.readNet(prototxt, weights)
    mask_net = load_model(model_path)
    return face_net, mask_net


def predict_frame(frame, face_net, mask_net, confidence=0.5):
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
        locs.append((startX, startY, endX, endY, float(conf)))

    preds = []
    if faces:
        batch = np.array(faces, dtype="float32")
        preds = mask_net.predict(batch, verbose=0)

    results = []
    annotated = frame.copy()
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
            "bbox": [int(startX), int(startY), int(endX), int(endY)],
        })

    return annotated, results


def process_image(image_path, face_net, mask_net, output_path, confidence=0.5):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    annotated, results = predict_frame(image, face_net, mask_net, confidence)
    if not results:
        raise RuntimeError(f"No faces detected in mask demo image: {image_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, annotated)
    return results


def process_video(video_path, face_net, mask_net, output_path, confidence=0.5,
                  max_seconds=12, sample_every=25):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = int(max_seconds * fps)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_idx = 0
    summary = {
        "frames_processed": 0,
        "frames_with_faces": 0,
        "sampled_detections": [],
    }

    while frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        annotated, results = predict_frame(frame, face_net, mask_net, confidence)
        writer.write(annotated)
        summary["frames_processed"] += 1
        if results:
            summary["frames_with_faces"] += 1
            if frame_idx % sample_every == 0:
                summary["sampled_detections"].append({"frame": frame_idx, "faces": results})
        frame_idx += 1

    cap.release()
    writer.release()

    if summary["frames_with_faces"] == 0:
        raise RuntimeError(f"No faces detected in mask demo video: {video_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Generate GitHub demo predictions")
    parser.add_argument("--samples-dir", default="github_demo", help="root folder for inputs/outputs")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    root = args.samples_dir
    input_img_dir = os.path.join(root, "inputs", "images")
    input_vid_dir = os.path.join(root, "inputs", "videos")
    output_img_dir = os.path.join(root, "outputs", "images")
    output_vid_dir = os.path.join(root, "outputs", "videos")
    report_path = os.path.join(root, "outputs", "predictions.json")

    print("[INFO] loading models...")
    face_net, mask_net = load_networks()

    report = {
        "task": "face_mask_detection",
        "not_face_recognition": True,
        "description": (
            "Face detection (SSD) + mask vs no-mask classification (MobileNetV2). "
            "All demo media shows people wearing or not wearing face masks. "
            "This is NOT face recognition — it does not identify who someone is."
        ),
        "images": [],
        "video": None,
    }

    for sample in PUBLIC_IMAGES:
        src_path = download_file(
            sample["url"],
            os.path.join(input_img_dir, sample["name"]),
            force=args.force_download,
        )
        stem, ext = os.path.splitext(sample["name"])
        out_path = os.path.join(output_img_dir, f"{stem}_predicted{ext}")
        print(f"[INFO] processing image {sample['name']}")
        detections = process_image(src_path, face_net, mask_net, out_path, args.confidence)
        report["images"].append({
            "input": src_path,
            "output": out_path,
            "description": sample["description"],
            "source": sample["source"],
            "license": sample["license"],
            "faces_detected": len(detections),
            "predictions": detections,
        })

    video_src = download_file(
        PUBLIC_VIDEO["url"],
        os.path.join(input_vid_dir, PUBLIC_VIDEO["name"]),
        force=args.force_download,
    )
    video_out = os.path.join(output_vid_dir, "people_masks_street_predicted.mp4")
    print(f"[INFO] processing video {PUBLIC_VIDEO['name']}")
    video_summary = process_video(video_src, face_net, mask_net, video_out, args.confidence)
    report["video"] = {
        "input": video_src,
        "output": video_out,
        "description": PUBLIC_VIDEO["description"],
        "source": PUBLIC_VIDEO["source"],
        "license": PUBLIC_VIDEO["license"],
        "page": PUBLIC_VIDEO["page"],
        **video_summary,
    }

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"[INFO] saved report to {report_path}")
    print("[INFO] done")


if __name__ == "__main__":
    main()
