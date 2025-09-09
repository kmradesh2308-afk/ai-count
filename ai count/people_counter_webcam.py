import argparse
import csv
import time
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(
        description="Real-time People Counter from webcam using YOLOv8"
    )
    p.add_argument("--source", type=str, default="0", help="Webcam index (e.g., 0,1,2)")
    p.add_argument("--weights", type=str, default="yolov8n.pt",
                   help="YOLOv8 weights: yolov8n.pt/yolov8s.pt/…")
    p.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.5, help="IoU threshold")
    p.add_argument("--device", type=str, default="cpu", help="cpu or CUDA index like 0")
    p.add_argument("--show", action="store_true", help="Show live window")
    p.add_argument("--save_csv", type=str, default="counts_log.csv",
                   help="CSV file to log (time_s,current_count)")
    p.add_argument("--track", action="store_true",
                   help="Use tracker for stable counts & unique-id stats")
    p.add_argument("--tracker", type=str, default="bytetrack.yaml",
                   help="Tracker config name (Ultralytics)")
    return p.parse_args()


def is_camera(src_str: str) -> bool:
    try:
        int(src_str)
        return True
    except ValueError:
        return False


def draw_hud(frame, present_count, fps, unique_total=None):
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = 10, 10, 10 + 330, 90
    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 0), -1)
    line1 = f"People in frame: {present_count}"
    line2 = f"FPS: {fps:.1f}"
    if unique_total is not None:
        line2 += f"  |  Unique seen: {unique_total}"
    cv2.putText(frame, line1, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, line2, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)


def main():
    args = parse_args()

    # Load model
    model = YOLO(args.weights)

    # Prepare CSV logging
    csv_file = None
    csv_writer = None
    if args.save_csv:
        csv_file = open(args.save_csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["time_s", "people_in_frame"])

    # Optional unique-id stats when tracking
    seen_ids = set()
    last_seen_frame = defaultdict(lambda: -10_000)

    # FPS measurement
    t_prev = time.time()
    fps_smooth = 0.0

    # Choose predict() or track()
    if args.track:
        # Tracking = more stable IDs; useful if you also want "unique seen"
        results_gen = model.track(
            source=int(args.source) if is_camera(args.source) else args.source,
            stream=True,
            persist=True,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            tracker=args.tracker,
            verbose=False,
        )
    else:
        # Plain detection; faster on weak CPUs
        results_gen = model.predict(
            source=int(args.source) if is_camera(args.source) else args.source,
            stream=True,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )

    try:
        for r in results_gen:
            frame = r.plot() if hasattr(r, "plot") else r.orig_img
            names = r.names

            # Count only class==person
            present_count = 0
            unique_total = None

            boxes = getattr(r, "boxes", None)
            if boxes is not None:
                clss = getattr(boxes, "cls", None)
                ids = getattr(boxes, "id", None)

                if clss is not None:
                    clss = clss.cpu().numpy().astype(int)
                    # Boolean mask for 'person' class
                    # COCO id for person is 0 for the standard YOLOv8 models
                    if isinstance(names, (list, dict)):
                        # Convert to names and check equality (safer with custom models)
                        present_count = sum(
                            1 for c in clss
                            if (names[c] if isinstance(names, list) else names.get(c, c)) == "person"
                        )
                    else:
                        # Fallback: assume class 0 is person
                        present_count = int((clss == 0).sum())

                # Track-based unique IDs
                if args.track and ids is not None and clss is not None:
                    ids_np = ids.cpu().numpy().astype(int)
                    for track_id, c in zip(ids_np, clss):
                        cname = names[c] if isinstance(names, (list, dict)) else str(c)
                        if cname == "person":
                            seen_ids.add(int(track_id))
                    unique_total = len(seen_ids)

            # FPS (EMA)
            t_now = time.time()
            inst = 1.0 / max((t_now - t_prev), 1e-6)
            fps_smooth = 0.9 * fps_smooth + 0.1 * inst if fps_smooth > 0 else inst
            t_prev = t_now

            # HUD overlay
            draw_hud(frame, present_count, fps_smooth, unique_total if args.track else None)

            # CSV log (once per frame)
            if csv_writer:
                csv_writer.writerow([round(t_now, 3), present_count])

            # Show window
            if args.show:
                cv2.imshow("People Counter (YOLOv8)", frame)
                # Press 'q' or ESC to quit
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    break

    except FileNotFoundError:
        print("\n[ERROR] Camera or source not found. Try a different --source index (0/1/2).")
    finally:
        if args.show:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        if csv_file:
            csv_file.close()
        print("\nStopped. If the camera didn’t open, close any app using it and retry with --source 1 or 2.")


if __name__ == "__main__":
    main()
