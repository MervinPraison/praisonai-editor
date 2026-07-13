"""Tests for audio volume measurement and normalisation helpers."""

from praisonai_editor.normalize import VolumeStats, _parse_volumedetect, needs_normalization


def test_parse_volumedetect():
    stderr = (
        "[Parsed_volumedetect_0] mean_volume: -24.5 dB\n"
        "[Parsed_volumedetect_0] max_volume: -3.2 dB\n"
    )
    stats = _parse_volumedetect(stderr)
    assert stats.mean_db == -24.5
    assert stats.max_db == -3.2


def test_needs_normalization_quiet_mean():
    stats = VolumeStats(mean_db=-25.0, max_db=-1.0)
    assert needs_normalization(stats) is True


def test_needs_normalization_quiet_peak():
    stats = VolumeStats(mean_db=-18.0, max_db=-10.0)
    assert needs_normalization(stats) is True


def test_needs_normalization_ok():
    stats = VolumeStats(mean_db=-20.0, max_db=-0.5)
    assert needs_normalization(stats) is False
