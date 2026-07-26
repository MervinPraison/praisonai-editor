"""Tests for manual time-range removal."""

import pytest

from praisonai_editor.remove_ranges import (
    build_remove_plan,
    parse_time,
    parse_time_range,
)


class TestParseTime:
    def test_seconds(self):
        assert parse_time("713") == 713.0
        assert parse_time(713.5) == 713.5

    def test_mm_ss(self):
        assert parse_time("11:53") == 11 * 60 + 53

    def test_hh_mm_ss(self):
        assert parse_time("1:11:53") == 3600 + 11 * 60 + 53


class TestParseTimeRange:
    def test_dash(self):
        assert parse_time_range("11:53-12:43") == (713.0, 763.0)

    def test_comma(self):
        assert parse_time_range("11:53,12:43") == (713.0, 763.0)

    def test_tuple(self):
        assert parse_time_range(("11:53", "12:43")) == (713.0, 763.0)

    def test_end_before_start(self):
        with pytest.raises(ValueError):
            parse_time_range("12:43-11:53")


class TestBuildRemovePlan:
    def test_single_middle_cut(self):
        plan = build_remove_plan(100.0, [(20.0, 30.0)])
        keep = plan.get_keep_segments()
        assert len(keep) == 2
        assert keep[0].start == 0.0 and keep[0].end == 20.0
        assert keep[1].start == 30.0 and keep[1].end == 100.0
        assert plan.removed_duration == pytest.approx(10.0)
        assert plan.edited_duration == pytest.approx(90.0)

    def test_overlapping_ranges_merge(self):
        plan = build_remove_plan(100.0, [(10.0, 25.0), (20.0, 40.0)])
        assert plan.removed_duration == pytest.approx(30.0)

    def test_range_beyond_duration(self):
        with pytest.raises(ValueError):
            build_remove_plan(60.0, [(0.0, 61.0)])
