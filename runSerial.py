# SERIALIZED VERSION
# SAFE TO RUN WHENEVER (AUTOSELECTS FREE-IST GPU)
import subprocess
import os
import cv2
import numpy as np
import motmetrics as mm

def get_freest_gpu():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,nounits,noheader"],
        capture_output=True, text=True
    )
    lines = result.stdout.strip().split("\n")
    gpu_memory = [(int(line.split(",")[0]), int(line.split(",")[1])) for line in lines]
    freest = max(gpu_memory, key=lambda x: x[1])
    print(f"Selected GPU {freest[0]} with {freest[1]} MB free")
    return freest[0]

gpu_id = get_freest_gpu()
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

from ultralytics import YOLO

MOT17_TRAIN = "./MOT17/train"
OUTPUT_DIR = "./results"

model = YOLO("yolo11n.pt")

def load_gt(gt_path):
    gt = {}
    with open(gt_path) as f:
        for line in f:
            parts = line.strip().split(",")
            frame, tid, x, y, w, h, conf, cls, vis = parts[:9]
            frame, tid = int(frame), int(tid)
            if int(cls) != 1 or float(vis) < 0.25:
                continue
            gt.setdefault(frame, []).append((tid, float(x), float(y), float(w), float(h)))
    return gt

def load_predictions(pred_path):
    preds = {}
    if not os.path.exists(pred_path):
        return preds
    with open(pred_path) as f:
        for line in f:
            parts = line.strip().split(",")
            frame, tid, x, y, w, h = parts[:6]
            frame, tid = int(frame), int(tid)
            preds.setdefault(frame, []).append((tid, float(x), float(y), float(w), float(h)))
    return preds

def iou_matrix(objs, hyps, max_iou=0.5):
    if np.size(objs) == 0 or np.size(hyps) == 0:
        return np.empty((0, 0))
    objs = np.asarray(objs, dtype=float)
    hyps = np.asarray(hyps, dtype=float)

    def to_xyxy(boxes):
        result = boxes.copy()
        result[:, 2] = boxes[:, 0] + boxes[:, 2]
        result[:, 3] = boxes[:, 1] + boxes[:, 3]
        return result

    objs_xyxy = to_xyxy(objs)
    hyps_xyxy = to_xyxy(hyps)

    distances = np.zeros((len(objs), len(hyps)))
    for i, o in enumerate(objs_xyxy):
        for j, h in enumerate(hyps_xyxy):
            ix1 = max(o[0], h[0])
            iy1 = max(o[1], h[1])
            ix2 = min(o[2], h[2])
            iy2 = min(o[3], h[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area_o = (o[2] - o[0]) * (o[3] - o[1])
            area_h = (h[2] - h[0]) * (h[3] - h[1])
            union = area_o + area_h - inter
            iou = inter / union if union > 0 else 0
            dist = 1 - iou
            distances[i, j] = dist if iou >= (1 - max_iou) else np.nan
    return distances

def evaluate_sequence(gt_path, pred_path):
    gt = load_gt(gt_path)
    preds = load_predictions(pred_path)
    acc = mm.MOTAccumulator(auto_id=True)

    all_frames = sorted(set(gt.keys()) | set(preds.keys()))
    for frame in all_frames:
        gt_dets = gt.get(frame, [])
        pred_dets = preds.get(frame, [])

        gt_ids = [d[0] for d in gt_dets]
        pred_ids = [d[0] for d in pred_dets]
        gt_boxes = np.array([[d[1], d[2], d[3], d[4]] for d in gt_dets]) if gt_dets else np.empty((0, 4))
        pred_boxes = np.array([[d[1], d[2], d[3], d[4]] for d in pred_dets]) if pred_dets else np.empty((0, 4))

        distances = iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
        acc.update(gt_ids, pred_ids, distances)
    return acc

def save_metrics(acc, sequence_name, output_folder):
    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=["num_frames", "mota", "motp", "idf1", "num_switches", "num_misses", "num_false_positives"],
        name=sequence_name
    )
    rendered = mm.io.render_summary(summary, namemap=mm.io.motchallenge_metric_names)

    metrics_path = os.path.join(output_folder, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Sequence: {sequence_name}\n")
        f.write("=" * 60 + "\n")
        f.write(rendered + "\n")
    print(f"  Metrics saved to {metrics_path}")

def frames_to_video(images_dir, output_path, fps=30):
    frames = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    if not frames:
        print(f"  No frames found in {images_dir}, skipping video creation.")
        return

    first = frames[0]
    ext = os.path.splitext(first)[1]
    digit_count = len(os.path.splitext(first)[0])
    pattern = os.path.join(images_dir, f"%0{digit_count}d{ext}")

    print(f"  Detected frame pattern: {pattern}")

    cmd = [
        os.path.expanduser("~/ffmpeg"), "-y",
        "-r", str(fps),
        "-i", pattern,
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"  Video saved to {output_path}")
    else:
        print(f"  ffmpeg error: {result.stderr}")

def get_fps_from_seqinfo(seq_dir):
    seqinfo_path = os.path.join(seq_dir, "seqinfo.ini")
    if os.path.exists(seqinfo_path):
        with open(seqinfo_path) as f:
            for line in f:
                if line.lower().startswith("framerate"):
                    return int(line.strip().split("=")[1])
    return 30

os.makedirs(OUTPUT_DIR, exist_ok=True)

all_accumulators = []
all_sequence_names = []

for sequence in sorted(os.listdir(MOT17_TRAIN)):
    img_dir = os.path.join(MOT17_TRAIN, sequence, "img1")
    gt_path = os.path.join(MOT17_TRAIN, sequence, "gt", "gt.txt")
    if not os.path.isdir(img_dir):
        continue

    seq_output_dir = os.path.abspath(os.path.join(OUTPUT_DIR, sequence))
    images_dir = os.path.abspath(os.path.join(seq_output_dir, "images"))
    os.makedirs(seq_output_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    pred_file_path = os.path.join(seq_output_dir, "predictions.txt")

    print(f"\nTracking {sequence}...")
    pred_file = open(pred_file_path, "w")
    frame_idx = 1

    results = model.track(
        source=img_dir,
        tracker="bytetrack.yaml",
        show=False,
        save=False,
        stream=True,
        device=0,
    )

    for result in results:
        # Save annotated frame with all detected classes drawn
        annotated = result.plot()
        frame_name = f"{frame_idx:06d}.jpg"
        cv2.imwrite(os.path.join(images_dir, frame_name), annotated)

        # Only write person detections (class 0) to predictions for MOT evaluation
        if result.boxes is not None and result.boxes.id is not None:
            for box, tid, cls in zip(
                result.boxes.xywh.cpu().numpy(),
                result.boxes.id.cpu().numpy(),
                result.boxes.cls.cpu().numpy()
            ):
                if int(cls) == 0:
                    x_center, y_center, w, h = box
                    x = x_center - w / 2
                    y = y_center - h / 2
                    pred_file.write(f"{frame_idx},{int(tid)},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")
        frame_idx += 1

    pred_file.close()
    print(f"  Saved {frame_idx - 1} annotated frames to {images_dir}")

    fps = get_fps_from_seqinfo(os.path.join(MOT17_TRAIN, sequence))
    video_path = os.path.join(seq_output_dir, f"{sequence}.mp4")
    print(f"  Creating video at {fps} fps...")
    frames_to_video(images_dir, video_path, fps=fps)

    if os.path.exists(gt_path):
        print(f"  Evaluating {sequence}...")
        acc = evaluate_sequence(gt_path, pred_file_path)
        save_metrics(acc, sequence, seq_output_dir)
        all_accumulators.append(acc)
        all_sequence_names.append(sequence)
    else:
        print(f"  No GT found for {sequence}, skipping evaluation.")

if all_accumulators:
    print("\nSaving overall summary...")
    mh = mm.metrics.create()
    summary = mh.compute_many(
        all_accumulators,
        metrics=["num_frames", "mota", "motp", "idf1", "num_switches", "num_misses", "num_false_positives"],
        names=all_sequence_names,
        generate_overall=True
    )
    rendered = mm.io.render_summary(summary, namemap=mm.io.motchallenge_metric_names)
    overall_path = os.path.join(OUTPUT_DIR, "overall_metrics.txt")
    with open(overall_path, "w") as f:
        f.write("Overall Metrics Across All Sequences\n")
        f.write("=" * 60 + "\n")
        f.write(rendered + "\n")
    print(f"Overall metrics saved to {overall_path}")
    print(rendered)

print("Done! Results saved to", OUTPUT_DIR)