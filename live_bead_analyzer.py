"""
live_bead_analyzer.py
Production 3D Concrete Printing Width Metrology Pipeline
Features:
 - Non-blocking background thread camera capture (zero USB buffer lag)
 - Real-time CSV quality control audit logging
 - Dynamic CLI arguments for nominal width, tolerances, and calibration ratio
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import threading
import time
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from ultralytics import YOLO

from bead_stabilizer import ProductionBeadStabilizer, get_adaptive_pixel_ratio


class ThreadedCamera:
    """Non-blocking camera thread to prevent frame backlog on USB action cameras."""

    def __init__(self, src: Union[int, str]):
        if isinstance(src, int):
            self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            if not self.cap.isOpened() and src == 1:
                self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(src)

        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame
            time.sleep(0.005)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.read_lock:
            return self.grabbed, self.frame.copy() if self.frame is not None else None

    def release(self):
        self.started = False
        if hasattr(self, "thread"):
            self.thread.join(timeout=1.0)
        self.cap.release()

    def get(self, prop_id):
        return self.cap.get(prop_id)

    def isOpened(self):
        return self.cap.isOpened()


# Constants
CLASS_COLORS = {0: (0, 255, 0), 1: (255, 165, 0), 2: (0, 140, 255), 3: (0, 0, 255)}
CLASS_NAMES = {0: "Normal", 1: "Overwidth", 2: "Smudged", 3: "Underwidth"}


def status_color(status: str) -> Tuple[int, int, int]:
    if status == "UNDER":
        return (0, 0, 255)
    if status == "OVER":
        return (255, 165, 0)
    if status == "NORMAL":
        return (0, 255, 0)
    return (160, 160, 160)


def choose_instance(result, frame_shape: Tuple[int, int]):
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return None, -1, None, 0.0

    h, w = frame_shape
    masks = result.masks.data.detach().cpu().numpy()
    cls_values = result.boxes.cls.detach().cpu().numpy().astype(int)
    conf_values = result.boxes.conf.detach().cpu().numpy()

    best, best_area = None, -1
    for i, raw_mask in enumerate(masks):
        arr = cv2.resize(raw_mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        binary = (arr > 0.5).astype(np.uint8)
        area = int(cv2.countNonZero(binary))
        if area > 250 and area > best_area:
            best_area = area
            best = (binary, int(cls_values[i]), None, float(conf_values[i]))
    return best if best is not None else (None, -1, None, 0.0)


def draw_dashboard(frame, measurements, fps, frame_index, nominal_w, tol):
    h, w = frame.shape[:2]
    x1, y1 = max(5, w - 320), 20
    x2, y2 = w - 10, y1 + 320

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (18, 22, 28), -1)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (90, 190, 230), 1)
    cv2.addWeighted(overlay, 0.86, frame, 0.14, 0, frame)

    cv2.putText(
        frame,
        f"3DCP METROLOGY (Tgt: {nominal_w:.0f}+-{tol:.0f}mm)",
        (x1 + 10, y1 + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (235, 245, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Frame {frame_index} | {fps:.1f} FPS",
        (x1 + 10, y1 + 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (180, 195, 210),
        1,
        cv2.LINE_AA,
    )

    by_node = {m["node"]: m for m in measurements}
    yy = y1 + 66
    for node in range(1, 11):
        m = by_node.get(node)
        val_text = f"{m['width_mm']:5.1f} mm" if (m and m.get("width_mm")) else "--.- mm"
        stat = m.get("status", "INVALID") if m else "INVALID"
        color = status_color(stat)

        cv2.putText(frame, f"W{node:<2}", (x1 + 10, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 225, 235), 1)
        cv2.putText(frame, val_text, (x1 + 80, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (240, 240, 240), 1)
        cv2.putText(frame, stat, (x1 + 190, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
        yy += 24


def main():
    parser = argparse.ArgumentParser(description="Production 3DCP Width Metrology")
    parser.add_argument("--model", default=r"D:\New width prj\runs\segment\train-6\weights\best.pt")
    parser.add_argument("--source", default=1)
    parser.add_argument("--output", default=r"D:\New width prj\live_production_output10.mp4")
    parser.add_argument("--ratio", type=float, default=0.320000)
    parser.add_argument("--nominal", type=float, default=50.0, help="Target bead nominal width in mm")
    parser.add_argument("--tol", type=float, default=10.0, help="Allowable width tolerance +- mm")
    parser.add_argument("--conf", type=float, default=0.22)
    parser.add_argument("--log-csv", default=r"D:\New width prj\metrology_telemetry.csv")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    model = YOLO(args.model)
    src_val = int(args.source) if str(args.source).isdigit() else args.source

    is_live = isinstance(src_val, int)
    cam = ThreadedCamera(src_val).start() if is_live else cv2.VideoCapture(src_val)

    width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cam.get(cv2.CAP_PROP_FPS)
    fps_in = 25.0 if (not np.isfinite(fps_in) or fps_in <= 0) else fps_in

    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (width, height))

    # Initialize CSV Log
    csv_file = open(args.log_csv, mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        ["Timestamp", "Frame", "FPS", "Mean_Width_mm", "Status", "W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "W9", "W10"]
    )

    stabilizer = ProductionBeadStabilizer(
        mask_ema_alpha=0.58,
        num_nodes=10,
        window_size=5,
        max_coast_frames=6,
        nominal_width_mm=args.nominal,
        tolerance_mm=args.tol,
    )

    frame_idx, last_t, fps_smooth = 0, time.perf_counter(), 0.0

    try:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                break
            frame_idx += 1

            frame_enh = stabilizer.preprocess_frame(frame)
            results = model.track(source=frame_enh, persist=True, retina_masks=True, imgsz=640, conf=args.conf, iou=0.45, verbose=False)
            res = results[0] if results else None

            binary = None
            if res is not None:
                binary, _, _, _ = choose_instance(res, frame.shape[:2])

            stable_binary = stabilizer.smooth_mask(binary)
            measurements = []
            spine = None

            if cv2.countNonZero(stable_binary) > 200:
                spine, measurements = stabilizer.compute_stable_slices(stable_binary, args.ratio)

                # Render Annotations
                tint = np.zeros_like(frame)
                tint[stable_binary > 0] = (0, 255, 0)
                frame = cv2.addWeighted(frame, 1.0, tint, 0.35, 0)

                if spine is not None and len(spine) >= 2:
                    cv2.polylines(frame, [np.round(spine).astype(np.intp)], False, (255, 255, 255), 2)

                for m in measurements:
                    if m.get("p1") and m.get("p2") and m.get("width_mm") is not None:
                        p1 = tuple(np.round(m["p1"]).astype(np.intp))
                        p2 = tuple(np.round(m["p2"]).astype(np.intp))
                        col = status_color(m["status"])

                        # 1. Draw the orthogonal slice line across the bead
                        cv2.line(frame, p1, p2, col, 2, cv2.LINE_AA)

                        # 2. Draw white endpoint anchor dots on the physical perimeter
                        cv2.circle(frame, p1, 3, (255, 255, 255), -1, cv2.LINE_AA)
                        cv2.circle(frame, p2, 3, (255, 255, 255), -1, cv2.LINE_AA)

                        # 3. Calculate midpoint near the centerline spine
                        mid_x = int((p1[0] + p2[0]) / 2)
                        mid_y = int((p1[1] + p2[1]) / 2)

                        # 4. Format label: e.g., "W1 49.3mm"
                        label_str = f"W{m['node']} {m['width_mm']:.1f}mm"

                        # 5. Draw high-contrast drop-shadow, then colored text
                        cv2.putText(
                            frame, label_str, (mid_x + 8, mid_y - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 2, cv2.LINE_AA
                        )
                        cv2.putText(
                            frame, label_str, (mid_x + 7, mid_y - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA
                        )

            now = time.perf_counter()
            dt = max(now - last_t, 1e-5)
            fps_smooth = (1.0 / dt) if fps_smooth <= 0 else (0.85 * fps_smooth + 0.15 * (1.0 / dt))
            last_t = now

            draw_dashboard(frame, measurements, fps_smooth, frame_idx, args.nominal, args.tol)
            writer.write(frame)

            # Log Telemetry
            valid_ws = [m["width_mm"] for m in measurements if m.get("width_mm")]
            mean_w = np.mean(valid_ws) if valid_ws else 0.0
            overall_status = "NORMAL" if (args.nominal - args.tol <= mean_w <= args.nominal + args.tol) else "DEFECT"
            node_vals = [f"{m['width_mm']:.1f}" if m.get("width_mm") is not None else "0.0" for m in measurements]
            while len(node_vals) < 10:
                node_vals.append("0.0")

            csv_writer.writerow(
                [datetime.now().strftime("%H:%M:%S.%f")[:-3], frame_idx, f"{fps_smooth:.1f}", f"{mean_w:.2f}", overall_status]
                + node_vals
            )

            if args.show:
                cv2.imshow("Industrial 3DCP Metrology", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

    finally:
        cam.release()
        writer.release()
        csv_file.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()