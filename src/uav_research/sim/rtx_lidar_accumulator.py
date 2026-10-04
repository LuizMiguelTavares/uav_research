"""Compatibility accumulator for the validated Isaac RTX OS1 profile."""

from __future__ import annotations

import numpy as np


class RtxLidarRevolutionAccumulator:
    """Assemble raw GMO segments between verified OS1 tick boundaries.

    This compatibility layer is validated for the
    ``OS1_REV6_32ch10hz512res`` profile with native RTX accumulation disabled.
    It deliberately uses that profile's ordered 511-to-0 tick wrap and must not
    be treated as a universal LiDAR accumulator.
    """

    def __init__(self) -> None:
        self.last_frame_id: int | None = None
        self.synchronized = False
        self.segments: list[np.ndarray] = []
        self.raw_segments_received = 0
        self.raw_scans_received = 0
        self.raw_scans_rejected = 0
        self.valid_scans = 0
        self.valid_point_min: int | None = None
        self.valid_point_max: int | None = None
        self.valid_point_sum = 0
        self.latest_valid_observation: np.ndarray | None = None
        self.latest_valid_timestamp_s: float | None = None
        self.latest_valid_age_s = float("inf")
        self.observation_valid = False
        self.new_scan = False

    @staticmethod
    def _xyz(gmo) -> np.ndarray:
        coordinates = np.column_stack((gmo.x, gmo.y, gmo.z)).astype(np.float32, copy=True)
        # GMO CoordsType: CARTESIAN=0, SPHERICAL=1.
        if int(gmo.elementsCoordsType) == 0:
            return coordinates

        azimuth = np.deg2rad(coordinates[:, 0])
        elevation = np.deg2rad(coordinates[:, 1])
        distance = coordinates[:, 2]
        horizontal_distance = distance * np.cos(elevation)
        return np.column_stack(
            (
                horizontal_distance * np.cos(azimuth),
                horizontal_distance * np.sin(azimuth),
                distance * np.sin(elevation),
            )
        ).astype(np.float32, copy=False)

    def _lose_synchronization(self) -> None:
        self.synchronized = False
        self.segments.clear()

    def update_age(self, simulation_time_s: float) -> None:
        """Update observation freshness from the current simulation time."""
        self.new_scan = False
        if self.latest_valid_timestamp_s is not None:
            self.latest_valid_age_s = max(0.0, simulation_time_s - self.latest_valid_timestamp_s)

    def add(self, gmo, *, timestamp_s: float | None = None) -> np.ndarray | None:
        """Consume one raw GMO segment and return one validated revolution."""
        frame_id = int(gmo.frameId)
        if frame_id == self.last_frame_id:
            return None

        frame_contiguous = self.last_frame_id is None or frame_id == self.last_frame_id + 1
        self.last_frame_id = frame_id
        self.raw_segments_received += 1
        if not frame_contiguous:
            self._lose_synchronization()

        points = self._xyz(gmo)
        ticks = np.asarray(gmo.tickId, dtype=np.int64).copy()
        scan_complete = bool(gmo.scanComplete)
        if len(points) != len(ticks):
            if scan_complete:
                self.raw_scans_received += 1
                self.raw_scans_rejected += 1
            self._lose_synchronization()
            return None

        wraps = np.flatnonzero(np.diff(ticks) < 0)
        has_verified_boundary = (
            scan_complete
            and len(wraps) == 1
            and ticks[wraps[0]] == 511
            and ticks[wraps[0] + 1] == 0
        )
        metadata_invalid = (scan_complete and not has_verified_boundary) or (not scan_complete and len(wraps) != 0)
        if metadata_invalid:
            if scan_complete:
                self.raw_scans_received += 1
                self.raw_scans_rejected += 1
            self._lose_synchronization()
            return None

        if not has_verified_boundary:
            if self.synchronized:
                self.segments.append(points)
            return None

        self.raw_scans_received += 1
        split = int(wraps[0]) + 1
        completed_scan = None
        if self.synchronized:
            completed_scan = np.concatenate((*self.segments, points[:split]))
        else:
            self.raw_scans_rejected += 1

        # The remainder belongs to the next revolution. This verified wrap is
        # also the synchronization point after startup or a dropped frame.
        self.segments = [points[split:]]
        self.synchronized = True

        if completed_scan is None or len(completed_scan) == 0:
            return None

        point_count = len(completed_scan)
        self.valid_scans += 1
        self.valid_point_sum += point_count
        self.valid_point_min = point_count if self.valid_point_min is None else min(self.valid_point_min, point_count)
        self.valid_point_max = point_count if self.valid_point_max is None else max(self.valid_point_max, point_count)
        self.latest_valid_observation = completed_scan
        self.latest_valid_timestamp_s = timestamp_s
        self.latest_valid_age_s = 0.0
        self.observation_valid = True
        self.new_scan = True
        return completed_scan
