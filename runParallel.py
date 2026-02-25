# PARALLELIZED VERSION
# DONT RUN UNLESS CLUSTER IS EMPTY
import subprocess
import os
import cv2
import numpy as np
import motmetrics as mm
import multiprocessing as mp
from multiprocessing import Pool

def get_free_gpus(max_gpus=3, min_free_mb=2000):
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,nounits,noheader"],
        capture_output=True, text=True
    )
    lines = result.stdout.strip().split("\n")
    gpu_memory = [(int(line.split(",")[0]), int(line.split(",")[1])) for line in lines]
    gpu_memory.sort(key=lambda x: x[1], reverse=True)
    selected = [idx for idx, free in gpu_memory if free >= min_free_mb][:max_gpus]
    if not selected:
        selected = [gpu_memory[0][0]]
    print(f"Selected GPUs: {selected}")
    return selected

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
    print(f"  [{sequence_name}] Metrics saved to {metrics_path}")

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

def process_sequence(args):
    """Worker function — runs in a separate process with its own CUDA context."""
    sequence, gpu_id, mot17_train, output_dir = args

    # Set GPU before importing torch/ultralytics
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from ultralytics import YOLO
    model = YOLO("yolo11n.pt")

    img_dir = os.path.join(mot17_train, sequence, "img1")
    gt_path = os.path.join(mot17_train, sequence, "gt", "gt.txt")

    seq_output_dir = os.path.abspath(os.path.join(output_dir, sequence))
    images_dir = os.path.abspath(os.path.join(seq_output_dir, "images"))
    os.makedirs(seq_output_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    pred_file_path = os.path.join(seq_output_dir, "predictions.txt")

    print(f"[GPU {gpu_id}] Tracking {sequence}...")
    pred_file = open(pred_file_path, "w")
    frame_idx = 1

    results = model.track(
        source=img_dir,
        tracker="bytetrack.yaml",
        show=False,
        save=False,
        stream=True,
        device=0,  # always 0 within this process since CUDA_VISIBLE_DEVICES is set
    )

    for result in results:
        annotated = result.plot()
        frame_name = f"{frame_idx:06d}.jpg"
        cv2.imwrite(os.path.join(images_dir, frame_name), annotated)

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
    print(f"[GPU {gpu_id}] {sequence}: saved {frame_idx - 1} frames")

    fps = get_fps_from_seqinfo(os.path.join(mot17_train, sequence))
    video_path = os.path.join(seq_output_dir, f"{sequence}.mp4")
    frames_to_video(images_dir, video_path, fps=fps)

    if os.path.exists(gt_path):
        print(f"[GPU {gpu_id}] Evaluating {sequence}...")
        acc = evaluate_sequence(gt_path, pred_file_path)
        save_metrics(acc, sequence, seq_output_dir)
        return sequence, acc
    else:
        print(f"[GPU {gpu_id}] No GT found for {sequence}, skipping evaluation.")
        return sequence, None


if __name__ == "__main__":
    MOT17_TRAIN = "./MOT17/train"
    OUTPUT_DIR = "./results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    gpu_ids = get_free_gpus(max_gpus=3, min_free_mb=2000)
    num_gpus = len(gpu_ids)

    sequences = sorted([
        s for s in os.listdir(MOT17_TRAIN)
        if os.path.isdir(os.path.join(MOT17_TRAIN, s, "img1"))
    ])

    # Assign GPUs round-robin across sequences
    tasks = [
        (seq, gpu_ids[i % num_gpus], MOT17_TRAIN, OUTPUT_DIR)
        for i, seq in enumerate(sequences)
    ]

    print(f"Running {len(sequences)} sequences across {num_gpus} GPUs ({num_gpus} at a time)...")

    # num_gpus workers so exactly that many sequences run simultaneously
    with Pool(processes=num_gpus, maxtasksperchild=1) as pool:
        results = pool.map(process_sequence, tasks)

    # Collect accumulators and compute overall metrics
    all_accumulators = []
    all_sequence_names = []
    for sequence, acc in results:
        if acc is not None:
            all_accumulators.append(acc)
            all_sequence_names.append(sequence)

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