from types import SimpleNamespace

import numpy as np
import pytest

from uav_research.sim.rtx_lidar_accumulator import RtxLidarRevolutionAccumulator


def make_gmo(
    frame_id: int,
    ticks: list[int],
    *,
    scan_complete: bool,
    point_count: int | None = None,
    coords_type: int = 0,
):
    count = len(ticks) if point_count is None else point_count
    values = np.arange(count, dtype=np.float32)
    return SimpleNamespace(
        frameId=frame_id,
        tickId=np.asarray(ticks, dtype=np.int64),
        scanComplete=scan_complete,
        elementsCoordsType=coords_type,
        x=values,
        y=values + 100.0,
        z=values + 200.0,
    )


@pytest.mark.unit
def test_startup_boundary_synchronizes_without_publishing() -> None:
    accumulator = RtxLidarRevolutionAccumulator()

    result = accumulator.add(make_gmo(1, [510, 511, 0, 1], scan_complete=True))

    assert result is None
    assert accumulator.synchronized
    assert accumulator.raw_scans_received == 1
    assert accumulator.raw_scans_rejected == 1
    assert accumulator.valid_scans == 0


@pytest.mark.unit
def test_segments_between_verified_boundaries_form_one_revolution() -> None:
    accumulator = RtxLidarRevolutionAccumulator()
    accumulator.add(make_gmo(1, [510, 511, 0, 1], scan_complete=True))
    accumulator.add(make_gmo(2, [2, 3], scan_complete=False))

    result = accumulator.add(make_gmo(3, [510, 511, 0, 1], scan_complete=True))

    assert result is not None
    assert result.shape == (6, 3)
    assert accumulator.raw_scans_received == 2
    assert accumulator.raw_scans_rejected == 1
    assert accumulator.valid_scans == 1
    assert accumulator.valid_point_min == 6
    assert accumulator.valid_point_max == 6


@pytest.mark.unit
def test_duplicate_frame_is_ignored() -> None:
    accumulator = RtxLidarRevolutionAccumulator()
    segment = make_gmo(1, [10, 11], scan_complete=False)
    accumulator.add(segment)

    assert accumulator.add(segment) is None
    assert accumulator.raw_segments_received == 1


@pytest.mark.unit
def test_frame_discontinuity_invalidates_current_revolution() -> None:
    accumulator = RtxLidarRevolutionAccumulator()
    accumulator.add(make_gmo(1, [511, 0], scan_complete=True))
    accumulator.add(make_gmo(2, [1, 2], scan_complete=False))
    accumulator.add(make_gmo(4, [3, 4], scan_complete=False))

    result = accumulator.add(make_gmo(5, [511, 0], scan_complete=True))

    assert result is None
    assert accumulator.synchronized
    assert accumulator.raw_scans_rejected == 2
    assert accumulator.valid_scans == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    ("ticks", "scan_complete"),
    [
        ([100, 101, 102], True),
        ([511, 0, 511, 0], True),
        ([511, 0], False),
    ],
)
def test_ambiguous_boundaries_are_rejected(ticks: list[int], scan_complete: bool) -> None:
    accumulator = RtxLidarRevolutionAccumulator()

    assert accumulator.add(make_gmo(1, ticks, scan_complete=scan_complete)) is None
    assert not accumulator.synchronized
    assert accumulator.valid_scans == 0


@pytest.mark.unit
def test_catastrophic_32_point_complete_output_is_not_accepted() -> None:
    accumulator = RtxLidarRevolutionAccumulator()

    result = accumulator.add(make_gmo(1, list(range(32)), scan_complete=True))

    assert result is None
    assert accumulator.raw_scans_received == 1
    assert accumulator.raw_scans_rejected == 1
    assert accumulator.valid_scans == 0


@pytest.mark.unit
def test_point_tick_length_mismatch_loses_synchronization() -> None:
    accumulator = RtxLidarRevolutionAccumulator()
    accumulator.add(make_gmo(1, [511, 0], scan_complete=True))

    result = accumulator.add(make_gmo(2, [1], point_count=2, scan_complete=True))

    assert result is None
    assert not accumulator.synchronized
    assert accumulator.raw_scans_rejected == 2


@pytest.mark.unit
def test_spherical_coordinates_are_converted_to_cartesian() -> None:
    accumulator = RtxLidarRevolutionAccumulator()
    gmo = make_gmo(1, [1], scan_complete=False, coords_type=1)
    gmo.x = np.asarray([0.0], dtype=np.float32)
    gmo.y = np.asarray([0.0], dtype=np.float32)
    gmo.z = np.asarray([2.0], dtype=np.float32)

    points = accumulator._xyz(gmo)

    np.testing.assert_allclose(points, [[2.0, 0.0, 0.0]], atol=1.0e-6)


@pytest.mark.unit
def test_latest_valid_observation_uses_simulation_time_without_ros_types() -> None:
    accumulator = RtxLidarRevolutionAccumulator()
    accumulator.add(make_gmo(1, [511, 0], scan_complete=True), timestamp_s=1.0)

    result = accumulator.add(make_gmo(2, [511, 0], scan_complete=True), timestamp_s=1.1)

    assert result is accumulator.latest_valid_observation
    assert accumulator.latest_valid_timestamp_s == pytest.approx(1.1)
    assert accumulator.latest_valid_age_s == 0.0
    assert accumulator.observation_valid
    assert accumulator.new_scan

    accumulator.update_age(1.15)
    assert accumulator.latest_valid_age_s == pytest.approx(0.05)
    assert not accumulator.new_scan
