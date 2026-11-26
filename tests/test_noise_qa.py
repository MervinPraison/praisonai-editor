"""Tests for background-noise QA classification."""

from praisonai_editor.noise_qa import (
    NoiseWindow,
    _estimate_noise_floor,
    _percentile,
    classify_noise,
)


def test_percentile():
    assert _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0) == 1.0
    assert _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50) == 3.0


def test_estimate_noise_floor_from_windows():
    windows = [
        NoiseWindow(0.0, -20.0, -10.0, -25.0, -30.0),
        NoiseWindow(0.5, -22.0, -12.0, -28.0, -32.0),
        NoiseWindow(1.0, -45.0, -35.0, -50.0, -55.0),
        NoiseWindow(1.5, -18.0, -8.0, -22.0, -28.0),
    ]
    nf, median, quiet = _estimate_noise_floor(windows)
    assert median == -21.0
    assert quiet < -40.0
    assert nf <= -50.0


def test_classify_clean():
    level, apply, nr, nf, tn, hp, reason = classify_noise(-55.0, 30.0)
    assert level == "clean"
    assert apply is False
    assert nr == 0.0
    assert hp is None
    assert "not needed" in reason


def test_classify_mild():
    level, apply, nr, nf, tn, hp, reason = classify_noise(-48.0, 24.0)
    assert level == "mild"
    assert apply is True
    assert nr == 10.0
    assert nf == -40.0
    assert hp is None


def test_classify_moderate():
    level, apply, nr, nf, tn, hp, reason = classify_noise(-38.0, 16.0, silence_ratio=0.2)
    assert level == "moderate"
    assert apply is True
    assert nr == 14.0
    assert nf == -33.0
    assert hp == 80


def test_classify_heavy():
    level, apply, nr, nf, tn, hp, reason = classify_noise(-30.0, 8.0, silence_ratio=0.2)
    assert level == "heavy"
    assert apply is True
    assert nr == 20.0
    assert hp == 80


def test_classify_constant_bed_mobile():
    level, apply, nr, nf, tn, hp, reason = classify_noise(-37.0, 16.0, silence_ratio=0.0)
    assert level == "moderate"
    assert apply is True
    assert nr >= 24.0
    assert tn is False
    assert hp == 120
    assert "Mobile" in reason or "continuous" in reason
