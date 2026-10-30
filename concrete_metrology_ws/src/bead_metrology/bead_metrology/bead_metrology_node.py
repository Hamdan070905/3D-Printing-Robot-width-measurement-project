#!/usr/bin/env python3
"""
bead_metrology_node.py
ROS 2 Python node for real-time 3D Concrete Printing (3DCP) bead width metrology,
defect classification, and monotonic scanline analysis.

Compatible with ROS 2 Humble / Iron / Jazzy.
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO


# Defect Classes
CLASS_NAMES: Dict[int, str] = {
    0: "Normal",
    1: "Overwidth",
    2: "Smughed",
    3: "Underwidth",
}

# Color schemes for overlays (BGR format)
CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (0, 255, 0),    # Normal: Green
    1: (255, 165, 0),  # Overwidth: Cyan/Blue-Orange (BGR)
    2: (0, 140, 255),  # Smughed: Orange
    3: (0, 0, 255),    # Underwidth: Red
}


class TemporalStabilizer:
    """Per-node rolling median filter to suppress high-frequency edge jitter."""
    def __init__(self, num_nodes: int = 10, buffer_size: int = 5):
        self.num_nodes = num_nodes
        self.buffer_size = buffer_size
        self.widths: Dict[int, Deque[float]] = {
            i: deque(maxlen=buffer_size) for i in range(1, num_nodes + 1)
        }
        self.endpoints: Dict[int, Deque[Tuple[float, float, float, float]]] = {
            i: deque(maxlen=buffer_size) for i in range(1, num_nodes + 1)
        }

    def update(self, measurements: List[dict], target_min_mm: float, target_max_mm: float) -> List[dict]:
        stabilized = []
        for m in measurements:
            item = dict(m)
            node = int(m["node"])
            if m["width_mm"] is not None and m["p1"] is not None and m["p2"] is not None:
                self.widths[node].append(float(m["width_mm"]))
                p1x, p1y = m["p1"]
                p2x, p2y = m["p2"]
                self.endpoints[node].append((p1x, p1y, p2x, p2y))

                width_vals = np.asarray(self.widths[node], dtype=np.float32)
                item["width_mm"] = float(np.median(width_vals))
                ep = np.asarray(self.endpoints[node], dtype=np.float32)
                ep_med = np.median(ep, axis=0)
                item["p1"] = (float(ep_med[0]), float(ep_med[1]))
                item["p2"] = (float(ep_med[2]), float(ep_med[3]))

                # Reclassify status based on stabilized median
                w_val = item["width_mm"]
                if w_val < target_min_mm:
                    item["status"] = "UNDER"
                elif w_val > target_max_mm:
                    item["status"] = "OVER"
                else:
                    item["status"] = "NORMAL"
            stabilized.append(item)
        return stabilized


class BeadMetrologyNode(Node):
    """
    ROS 2 Node that wraps YOLOv11n-seg for concrete bead segmentation,
    monotonic horizontal scanline width slicing, and defect telemetry publication.
    """

    def __init__(self):
        super().__init__('bead_metrology_node')
        self.get_logger().info("Initializing 3DCP Bead Metrology ROS 2 Node...")

        # ------------------- ROS 2 Parameters -------------------
        self.declare_parameter('model_path', 'weights/best.pt')
        self.declare_parameter('inference_size', 640)
        self.declare_parameter('confidence_threshold', 0.25)
        self.declare_parameter('iou_threshold', 0.50)

        self.declare_parameter('pixel_to_mm_ratio', 0.32)
        self.declare_parameter('target_min_mm', 40.0)
        self.declare_parameter('target_max_mm', 60.0)
        self.declare_parameter('valid_slice_min_mm', 10.0)
        self.declare_parameter('valid_slice_max_mm', 120.0)
        self.declare_parameter('num_width_nodes', 10)
        self.declare_parameter('temporal_buffer_size', 5)

        self.declare_parameter('camera_topic', '/gantry_cam/image_raw')
        self.declare_parameter('telemetry_topic', '/3dcp/metrology_telemetry')
        self.declare_parameter('annotated_image_topic', '/3dcp/annotated_image')
        self.declare_parameter('publish_annotated_image', True)
        self.declare_parameter('frame_drop_warn_threshold', 5)

        # Retrieve parameters
        raw_model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.inference_size = self.get_parameter('inference_size').get_parameter_value().integer_value
        self.conf_thresh = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.iou_thresh = self.get_parameter('iou_threshold').get_parameter_value().double_value

        self.ratio_mm_per_px = self.get_parameter('pixel_to_mm_ratio').get_parameter_value().double_value
        self.target_min_mm = self.get_parameter('target_min_mm').get_parameter_value().double_value
        self.target_max_mm = self.get_parameter('target_max_mm').get_parameter_value().double_value
        self.valid_min_mm = self.get_parameter('valid_slice_min_mm').get_parameter_value().double_value
        self.valid_max_mm = self.get_parameter('valid_slice_max_mm').get_parameter_value().double_value
        self.num_nodes = self.get_parameter('num_width_nodes').get_parameter_value().integer_value
        self.buffer_size = self.get_parameter('temporal_buffer_size').get_parameter_value().integer_value

        self.camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        self.telemetry_topic = self.get_parameter('telemetry_topic').get_parameter_value().string_value
        self.annotated_topic = self.get_parameter('annotated_image_topic').get_parameter_value().string_value
        self.publish_annotated = self.get_parameter('publish_annotated_image').get_parameter_value().bool_value
        self.frame_drop_thresh = self.get_parameter('frame_drop_warn_threshold').get_parameter_value().integer_value

        # Cross-platform model resolution
        self.model_path = Path(raw_model_path).expanduser().resolve()
        if not self.model_path.exists():
            self.get_logger().error(f"YOLO weights not found at path: {self.model_path}")
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")

        self.get_logger().info(f"Loading YOLOv11n-seg model from: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        self.get_logger().info("Model successfully loaded into inference engine.")

        # Metrology components
        self.stabilizer = TemporalStabilizer(num_nodes=self.num_nodes, buffer_size=self.buffer_size)
        self.bridge = CvBridge()

        # Telemetry metrics
        self.frame_count = 0
        self.last_frame_time = time.perf_counter()
        self.fps_smooth = 0.0
        self.missed_detections = 0

        # ------------------- ROS 2 Pub / Sub -------------------
        # Camera subscriber with SensorData QoS (best-effort / volatile for real-time video)
        qos_camera = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.sub_cam = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            qos_camera
        )

        # Telemetry publisher
        self.pub_telemetry = self.create_publisher(String, self.telemetry_topic, 10)

        # Annotated visualization image publisher (optional)
        if self.publish_annotated:
            self.pub_annotated = self.create_publisher(Image, self.annotated_topic, 10)

        self.get_logger().info(
            f"Node online. Subscribed to '{self.camera_topic}'. "
            f"Broadcasting telemetry on '{self.telemetry_topic}'. "
            f"Calibration: {self.ratio_mm_per_px:.4f} mm/px"
        )

    def image_callback(self, msg: Image) -> None:
        """Process incoming camera frame and execute metrology pipeline."""
        start_t = time.perf_counter()
        self.frame_count += 1

        try:
            cv_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge conversion error: {e}")
            return

        if cv_frame is None or cv_frame.size == 0:
            self.get_logger().warn("Received empty camera frame, skipping.")
            return

        h, w = cv_frame.shape[:2]

        # YOLO inference
        results = self.model.track(
            source=cv_frame,
            persist=True,
            retina_masks=True,
            agnostic_nms=True,
            imgsz=self.inference_size,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            verbose=False
        )

        result = results[0] if results else None
        binary_mask, class_id, track_id, confidence = self._choose_instance(result, (h, w))

        measurements: List[dict] = []
        spine = None

        if binary_mask is not None:
            self.missed_detections = 0
            spine, raw_measurements = self._compute_width_slices(binary_mask)
            measurements = self.stabilizer.update(raw_measurements, self.target_min_mm, self.target_max_mm)
        else:
            self.missed_detections += 1
            if self.missed_detections % self.frame_drop_thresh == 0:
                self.get_logger().warn(
                    f"No bead segmentation mask detected for {self.missed_detections} consecutive frames!"
                )

        # Calculate processing FPS and execution latency
        now = time.perf_counter()
        dt = max(now - self.last_frame_time, 1e-6)
        instant_fps = 1.0 / dt
        self.fps_smooth = instant_fps if self.fps_smooth <= 0 else 0.85 * self.fps_smooth + 0.15 * instant_fps
        self.last_frame_time = now
        latency_ms = (now - start_t) * 1000.0

        # Construct structured telemetry payload
        telemetry_payload = self._build_telemetry(
            msg.header.stamp,
            class_id,
            track_id,
            confidence,
            measurements,
            latency_ms
        )

        # Publish telemetry JSON string to PLC / UI topic
        str_msg = String()
        str_msg.data = json.dumps(telemetry_payload)
        self.pub_telemetry.publish(str_msg)

        # Publish annotated debug frame if enabled
        if self.publish_annotated:
            annotated_frame = self._render_annotation(
                cv_frame, binary_mask, class_id, spine, measurements, confidence, track_id
            )
            try:
                annotated_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
                annotated_msg.header = msg.header
                self.pub_annotated.publish(annotated_msg)
            except CvBridgeError as e:
                self.get_logger().error(f"Failed to publish annotated image: {e}")

    def _choose_instance(
        self, result, frame_shape: Tuple[int, int]
    ) -> Tuple[Optional[np.ndarray], int, Optional[int], float]:
        """Extract dominant bead mask from inference results."""
        if result is None or result.masks is None or result.boxes is None or len(result.boxes) == 0:
            return None, -1, None, 0.0

        h, w = frame_shape
        masks = result.masks.data.detach().cpu().numpy()
        cls_values = result.boxes.cls.detach().cpu().numpy().astype(int)
        conf_values = result.boxes.conf.detach().cpu().numpy()
        track_ids = None
        if getattr(result.boxes, "id", None) is not None:
            try:
                track_ids = result.boxes.id.detach().cpu().numpy().astype(int)
            except Exception:
                track_ids = None

        best = None
        best_area = -1
        min_area = 250

        for i, raw_mask in enumerate(masks):
            arr = np.asarray(raw_mask)
            if arr.shape[:2] != (h, w):
                arr = cv2.resize(arr.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            binary = (arr > 0.5).astype(np.uint8)
            area = int(cv2.countNonZero(binary))
            if area < min_area:
                continue
            if area > best_area:
                best_area = area
                tid = int(track_ids[i]) if track_ids is not None and i < len(track_ids) else None
                best = (binary, int(cls_values[i]), tid, float(conf_values[i]))

        return best if best is not None else (None, -1, None, 0.0)

    def _compute_width_slices(self, binary: np.ndarray) -> Tuple[Optional[np.ndarray], List[dict]]:
        """
        Monotonic horizontal scanline slicing:
        Generates 10 strictly ordered Y-intervals from top to bottom.
        Guarantees zero line crossings, monotonic W1 -> W10 order, and clean medial axis.
        """
        ys, xs = np.where(binary > 0)
        if len(xs) < 50:
            return None, []

        y_min, y_max = int(ys.min()), int(ys.max())
        height_span = y_max - y_min
        if height_span < 30:
            return None, []

        y_coords = np.linspace(y_min + height_span * 0.08, y_max - height_span * 0.08, self.num_nodes)
        measurements: List[dict] = []
        spine_pts = []
        w_cols = binary.shape[1]

        for node_num, y_val in enumerate(y_coords, start=1):
            row = int(round(y_val))
            if row < 0 or row >= binary.shape[0]:
                continue

            row_pixels = np.where(binary[row, :] > 0)[0]
            if len(row_pixels) < 2:
                measurements.append({
                    "node": node_num, "center": (w_cols // 2, row),
                    "p1": None, "p2": None, "width_mm": None, "status": "INVALID"
                })
                continue

            x_left = float(row_pixels[0])
            x_right = float(row_pixels[-1])
            x_mid = (x_left + x_right) / 2.0

            p1 = (x_left, float(row))
            p2 = (x_right, float(row))
            center = (x_mid, float(row))
            spine_pts.append(center)

            width_px = x_right - x_left
            width_mm = width_px * self.ratio_mm_per_px

            if not np.isfinite(width_mm) or not (self.valid_min_mm <= width_mm <= self.valid_max_mm):
                measurements.append({
                    "node": node_num, "center": center, "p1": None, "p2": None,
                    "width_mm": None, "status": "INVALID"
                })
                continue

            if width_mm < self.target_min_mm:
                status = "UNDER"
            elif width_mm > self.target_max_mm:
                status = "OVER"
            else:
                status = "NORMAL"

            measurements.append({
                "node": node_num,
                "center": center,
                "p1": p1,
                "p2": p2,
                "width_mm": float(width_mm),
                "status": status
            })

        spine = np.array(spine_pts, dtype=np.float32) if spine_pts else None
        return spine, measurements

    def _build_telemetry(
        self,
        stamp,
        class_id: int,
        track_id: Optional[int],
        confidence: float,
        measurements: List[dict],
        latency_ms: float
    ) -> dict:
        """Construct JSON-serializable telemetry message for gantry PLC / HMI."""
        valid_widths = [m["width_mm"] for m in measurements if m.get("width_mm") is not None]
        mean_width = float(np.mean(valid_widths)) if valid_widths else None
        std_width = float(np.std(valid_widths)) if len(valid_widths) > 1 else 0.0

        under_count = sum(1 for m in measurements if m.get("status") == "UNDER")
        over_count = sum(1 for m in measurements if m.get("status") == "OVER")
        invalid_count = sum(1 for m in measurements if m.get("status") == "INVALID")

        defect_alert = "NONE"
        if over_count > 0:
            defect_alert = "OVERWIDTH_WARNING"
        elif under_count > 0:
            defect_alert = "UNDERWIDTH_WARNING"
        elif class_id == 2:
            defect_alert = "SMUGHED_WARNING"

        return {
            "header": {
                "frame_id": "gantry_cam_optical_frame",
                "stamp_sec": stamp.sec,
                "stamp_nanosec": stamp.nanosec,
                "frame_index": self.frame_count,
            },
            "classification": {
                "class_id": class_id,
                "class_name": CLASS_NAMES.get(class_id, "Unknown"),
                "confidence": round(confidence, 4),
                "track_id": track_id
            },
            "metrology": {
                "mean_width_mm": round(mean_width, 2) if mean_width is not None else None,
                "std_width_mm": round(std_width, 2),
                "valid_slices": len(valid_widths),
                "total_slices": self.num_nodes,
                "defect_alert": defect_alert,
                "slices": [
                    {
                        "node": m["node"],
                        "width_mm": round(m["width_mm"], 2) if m.get("width_mm") is not None else None,
                        "status": m.get("status", "INVALID")
                    }
                    for m in measurements
                ]
            },
            "performance": {
                "inference_fps": round(self.fps_smooth, 1),
                "latency_ms": round(latency_ms, 2)
            }
        }

    def _render_annotation(
        self,
        frame: np.ndarray,
        binary: Optional[np.ndarray],
        class_id: int,
        spine: Optional[np.ndarray],
        measurements: List[dict],
        confidence: float,
        track_id: Optional[int]
    ) -> np.ndarray:
        """Render metrology dashboard and scanlines on frame for visual telemetry."""
        output = frame.copy()
        if binary is not None:
            # Mask overlay
            color = CLASS_COLORS.get(class_id, (255, 255, 255))
            tint = np.zeros_like(output)
            tint[binary > 0] = color
            blended = cv2.addWeighted(output, 1.0, tint, 0.38, 0)
            output[binary > 0] = blended[binary > 0]

            # White medial axis
            if spine is not None and len(spine) >= 2:
                pts = np.round(spine).astype(np.intp).reshape(-1, 1, 2)
                cv2.polylines(output, [pts], False, (255, 255, 255), 2, cv2.LINE_AA)

            # Monotonic Scanlines
            for m in measurements:
                if m.get("width_mm") is None or m.get("p1") is None or m.get("p2") is None:
                    continue
                p1 = tuple(np.round(m["p1"]).astype(np.intp))
                p2 = tuple(np.round(m["p2"]).astype(np.intp))
                status = m.get("status", "INVALID")

                if status == "UNDER":
                    line_color = (0, 0, 255)
                elif status == "OVER":
                    line_color = (255, 165, 0)
                elif status == "NORMAL":
                    line_color = (0, 255, 0)
                else:
                    line_color = (160, 160, 160)

                cv2.line(output, p1, p2, line_color, 2, cv2.LINE_AA)
                cv2.circle(output, p1, 3, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(output, p2, 3, (255, 255, 255), -1, cv2.LINE_AA)
                mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                cv2.putText(output, f"W{m['node']} {m['width_mm']:.1f}mm", (mid[0] + 5, mid[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, line_color, 1, cv2.LINE_AA)

        # Status text overlay
        cv2.putText(output, f"ROS 2 3DCP | FPS: {self.fps_smooth:.1f} | Conf: {confidence:.2f}",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        return output


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = BeadMetrologyNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[FATAL] Bead metrology node crashed: {e}")
    finally:
        if node is not None:
            node.get_logger().info("Shutting down Bead Metrology ROS 2 Node...")
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
