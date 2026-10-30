"""
bead_stabilizer.py - Industrial Grade Stabilizer for 3DCP Metrology
Hardened Features:
 - Dynamic nominal width & tolerance checking (no hardcoded limits)
 - Adaptive PCA with local tangent fallback for curved beads
 - Sub-pixel parabolic distance transform ridge extraction
 - 6-frame coasting hold with 0.45 threshold
"""

from collections import deque
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple


def get_adaptive_pixel_ratio(frame_width: int, baseline_ratio: float = 0.32, baseline_width: int = 854) -> float:
    return baseline_ratio * (baseline_width / float(frame_width))


class ProductionBeadStabilizer:
    def __init__(
        self,
        mask_ema_alpha: float = 0.58,
        num_nodes: int = 10,
        window_size: int = 5,
        min_width_mm: float = 8.0,
        max_width_mm: float = 120.0,
        max_coast_frames: int = 6,
        nominal_width_mm: float = 50.0,
        tolerance_mm: float = 10.0,
    ):
        self.alpha = mask_ema_alpha
        self.num_nodes = num_nodes
        self.window_size = window_size
        self.min_w_mm = min_width_mm
        self.max_w_mm = max_width_mm
        self.max_coast = max_coast_frames
        self.nominal_w = nominal_width_mm
        self.tolerance = tolerance_mm

        self.prev_prob_mask: Optional[np.ndarray] = None
        self.coast_counter = 0
        self.prev_major_axis: Optional[np.ndarray] = None

        self.clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        self.width_history: Dict[int, deque] = {
            i: deque(maxlen=window_size) for i in range(1, num_nodes + 1)
        }
        self.p1_history: Dict[int, deque] = {
            i: deque(maxlen=window_size) for i in range(1, num_nodes + 1)
        }
        self.p2_history: Dict[int, deque] = {
            i: deque(maxlen=window_size) for i in range(1, num_nodes + 1)
        }

    def preprocess_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = self.clahe.apply(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    def smooth_mask(self, raw_binary: Optional[np.ndarray]) -> np.ndarray:
        if raw_binary is not None and cv2.countNonZero(raw_binary) > 200:
            curr_prob = raw_binary.astype(np.float32)
            self.coast_counter = 0
            if self.prev_prob_mask is None or self.prev_prob_mask.shape != curr_prob.shape:
                self.prev_prob_mask = curr_prob.copy()
            else:
                self.prev_prob_mask = (self.alpha * curr_prob) + ((1.0 - self.alpha) * self.prev_prob_mask)
        else:
            self.coast_counter += 1
            if self.prev_prob_mask is not None and self.coast_counter <= self.max_coast:
                self.prev_prob_mask *= 0.88
            else:
                self.prev_prob_mask = None
                self.prev_major_axis = None
                return np.zeros((100, 100), dtype=np.uint8) if raw_binary is None else np.zeros_like(raw_binary)

        stable_binary = (self.prev_prob_mask > 0.45).astype(np.uint8)
        stable_binary = cv2.morphologyEx(stable_binary, cv2.MORPH_CLOSE, self.morph_kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(stable_binary, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            return (labels == largest_label).astype(np.uint8)
        return stable_binary

    def compute_stable_slices(
        self, binary: np.ndarray, ratio_mm_per_px: float
    ) -> Tuple[Optional[np.ndarray], List[dict]]:
        if binary is None or cv2.countNonZero(binary) < 300:
            return None, []

        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        if dist_transform.max() < 4.0:
            return None, []

        ys, xs = np.where(binary > 0)
        pts_all = np.column_stack([xs, ys]).astype(np.float32)

        mean, eigenvectors = cv2.PCACompute(pts_all, mean=None)
        raw_major = eigenvectors[0].copy()

        if self.prev_major_axis is not None:
            if np.dot(raw_major, self.prev_major_axis) < 0.0:
                raw_major = -raw_major

            dot_val = np.clip(np.dot(raw_major, self.prev_major_axis), 0.0, 1.0)
            adaptive_beta = 0.35 + 0.35 * (1.0 - dot_val)
            blended = (1.0 - adaptive_beta) * self.prev_major_axis + adaptive_beta * raw_major
            norm_b = np.linalg.norm(blended)
            major_axis = blended / (norm_b if norm_b > 1e-6 else 1.0)
        else:
            major_axis = raw_major

        self.prev_major_axis = major_axis.copy()

        projections = np.dot(pts_all - mean, major_axis)
        p_min, p_max = float(projections.min()), float(projections.max())
        span = p_max - p_min

        if span < 30.0:
            return None, []

        sample_projections = np.linspace(p_min + span * 0.08, p_max - span * 0.08, num=40)
        ortho_axis = np.array([-major_axis[1], major_axis[0]], dtype=np.float32)

        raw_spine_pts = []
        h, w = binary.shape[:2]
        offsets = np.linspace(-60.0, 60.0, num=41)
        step_sz = offsets[1] - offsets[0]

        for s_proj in sample_projections:
            center_sample = mean[0] + major_axis * s_proj
            dist_profile = []

            for off in offsets:
                tpt = center_sample + ortho_axis * off
                ix, iy = int(round(tpt[0])), int(round(tpt[1]))
                if 0 <= ix < w and 0 <= iy < h and binary[iy, ix] > 0:
                    dist_profile.append(float(dist_transform[iy, ix]))
                else:
                    dist_profile.append(-1.0)

            if len(dist_profile) >= 3:
                k_max = int(np.argmax(dist_profile))
                if dist_profile[k_max] > 1.0:
                    best_off = offsets[k_max]
                    if 0 < k_max < len(dist_profile) - 1:
                        d_prev = dist_profile[k_max - 1]
                        d_mid = dist_profile[k_max]
                        d_next = dist_profile[k_max + 1]
                        denom = d_prev - 2.0 * d_mid + d_next
                        if abs(denom) > 1e-4:
                            delta_k = (d_prev - d_next) / (2.0 * denom)
                            if abs(delta_k) < 1.0:
                                best_off += delta_k * step_sz

                    pt = center_sample + ortho_axis * best_off
                    raw_spine_pts.append((float(pt[0]), float(pt[1])))

        if len(raw_spine_pts) < 6:
            return None, []

        pts = np.array(raw_spine_pts, dtype=np.float32)
        smooth_x = np.convolve(pts[:, 0], np.ones(5) / 5.0, mode="valid")
        smooth_y = np.convolve(pts[:, 1], np.ones(5) / 5.0, mode="valid")
        smoothed_spine = np.column_stack([smooth_x, smooth_y])

        if len(smoothed_spine) < self.num_nodes:
            return smoothed_spine, []

        diffs = np.diff(smoothed_spine, axis=0)
        seg_lens = np.sqrt((diffs**2).sum(axis=1))
        cum_len = np.insert(np.cumsum(seg_lens), 0, 0.0)
        total_len = cum_len[-1]

        if total_len < 20.0:
            return smoothed_spine, []

        node_distances = np.linspace(total_len * 0.10, total_len * 0.90, self.num_nodes)
        measurements = []

        for i, d_target in enumerate(node_distances):
            node_id = i + 1
            idx = np.searchsorted(cum_len, d_target)
            idx = np.clip(idx, 1, len(smoothed_spine) - 1)
            t = (d_target - cum_len[idx - 1]) / max(cum_len[idx] - cum_len[idx - 1], 1e-5)
            center = (1.0 - t) * smoothed_spine[idx - 1] + t * smoothed_spine[idx]
            cx, cy = float(center[0]), float(center[1])

            p_prev = smoothed_spine[max(0, idx - 2)]
            p_next = smoothed_spine[min(len(smoothed_spine) - 1, idx + 2)]
            tx = p_next[0] - p_prev[0]
            ty = p_next[1] - p_prev[1]
            tangent_len = np.hypot(tx, ty)

            if tangent_len < 1e-4:
                continue

            nx = -ty / tangent_len
            ny = tx / tangent_len

            p1 = None
            for step in np.arange(1.0, 160.0, 1.0):
                rx = int(round(cx + nx * step))
                ry = int(round(cy + ny * step))
                if rx < 0 or rx >= w or ry < 0 or ry >= h or binary[ry, rx] == 0:
                    p1 = (cx + nx * (step - 0.5), cy + ny * (step - 0.5))
                    break

            p2 = None
            for step in np.arange(1.0, 160.0, 1.0):
                rx = int(round(cx - nx * step))
                ry = int(round(cy - ny * step))
                if rx < 0 or rx >= w or ry < 0 or ry >= h or binary[ry, rx] == 0:
                    p2 = (cx - nx * (step - 0.5), cy - ny * (step - 0.5))
                    break

            if p1 is not None and p2 is not None:
                measured_width_px = np.hypot(p1[0] - p2[0], p1[1] - p2[1])
                width_mm = measured_width_px * ratio_mm_per_px

                if self.min_w_mm <= width_mm <= self.max_w_mm:
                    self.width_history[node_id].append(width_mm)
                    self.p1_history[node_id].append(p1)
                    self.p2_history[node_id].append(p2)

                    stable_w = float(np.median(self.width_history[node_id]))
                    med_p1 = tuple(np.median(np.array(self.p1_history[node_id]), axis=0))
                    med_p2 = tuple(np.median(np.array(self.p2_history[node_id]), axis=0))
                    med_center = ((med_p1[0] + med_p2[0]) / 2.0, (med_p1[1] + med_p2[1]) / 2.0)

                    # Dynamic tolerance classification
                    lower_limit = self.nominal_w - self.tolerance
                    upper_limit = self.nominal_w + self.tolerance
                    status = "UNDER" if stable_w < lower_limit else ("OVER" if stable_w > upper_limit else "NORMAL")

                    measurements.append({
                        "node": node_id,
                        "center": med_center,
                        "p1": med_p1,
                        "p2": med_p2,
                        "width_mm": stable_w,
                        "status": status,
                    })
                    continue

            measurements.append({
                "node": node_id,
                "center": (cx, cy),
                "p1": None,
                "p2": None,
                "width_mm": None,
                "status": "INVALID",
            })

        return smoothed_spine, measurements